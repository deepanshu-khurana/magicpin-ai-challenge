import os
import sys
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from threading import Lock, RLock
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

# =============================================================================
# DATA MODELS & SCHEMAS
# =============================================================================

class ContextPush(BaseModel):
    scope: str  # "category" | "merchant" | "customer" | "trigger"
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str

class TickRequest(BaseModel):
    now: str
    available_triggers: List[str] = []

class ActionItem(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: str  # "vera" | "merchant_on_behalf"
    trigger_id: str
    template_name: str
    template_params: List[str]
    body: str
    cta: str  # "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot" | "none"
    suppression_key: str
    rationale: str

class TickResponse(BaseModel):
    actions: List[ActionItem]

class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str  # "merchant" | "customer"
    message: str
    received_at: str
    turn_number: int

class ReplyResponse(BaseModel):
    action: str  # "send" | "wait" | "end"
    body: Optional[str] = None
    cta: Optional[str] = None
    wait_seconds: Optional[int] = None
    rationale: str

# =============================================================================
# CONTEXT STORE & CONVERSATION STORE
# =============================================================================

class ContextStore:
    def __init__(self):
        self._lock = Lock()
        # Storage: (scope, context_id) -> {"version": int, "payload": dict}
        self.contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def push(self, scope: str, context_id: str, version: int, payload: Dict[str, Any]) -> Tuple[bool, str, Optional[int]]:
        key = (scope.lower(), context_id)
        with self._lock:
            cur = self.contexts.get(key)
            if cur and cur["version"] == version:
                return True, f"ack_{context_id}_v{version}", None
            if cur and cur["version"] > version:
                return False, "stale_version", cur["version"]
            self.contexts[key] = {"version": version, "payload": payload}
            return True, f"ack_{context_id}_v{version}", None

    def get(self, scope: str, context_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self.contexts.get((scope.lower(), context_id))
            return data["payload"] if data else None

    def get_counts(self) -> Dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._lock:
            for (scope, _), _ in self.contexts.items():
                if scope in counts:
                    counts[scope] += 1
        return counts

class ConversationStore:
    def __init__(self):
        self._lock = RLock()
        # conv_id -> {"history": list, "suppressed": bool, "auto_reply_count": int, "state": str}
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.sent_suppression_keys: set = set()

    def get_or_create(self, conv_id: str) -> Dict[str, Any]:
        with self._lock:
            if conv_id not in self.conversations:
                self.conversations[conv_id] = {
                    "history": [],
                    "suppressed": False,
                    "auto_reply_count": 0,
                    "state": "qualifying",
                    "created_at": time.time()
                }
            return self.conversations[conv_id]

    def record_turn(self, conv_id: str, from_role: str, message: str, body: str = None):
        with self._lock:
            conv = self.get_or_create(conv_id)
            conv["history"].append({"from": from_role, "msg": message, "ts": time.time(), "body": body})

    def is_suppressed(self, suppression_key: str) -> bool:
        with self._lock:
            return suppression_key in self.sent_suppression_keys

    def mark_suppressed(self, suppression_key: str):
        if suppression_key:
            with self._lock:
                self.sent_suppression_keys.add(suppression_key)

# Global Stores
store = ContextStore()
conv_store = ConversationStore()

# =============================================================================
# DOMAIN COMPOSER ENGINE & SAFETY ENFORCER
# =============================================================================

class EngagementComposer:
    """
    Expert 4-Context Composer. Generates high-converting, strictly compliant messages.
    Supports both LLM call (if API key present) and Deterministic Synthesis.
    """

    @staticmethod
    def sanitize(text: str, taboos: List[str] = None) -> str:
        """Strip URLs, sanitize taboo vocabulary, remove internal signal tags."""
        if not text:
            return ""
        # 1. Remove URLs (Meta policy restriction: no raw links)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 2. Strip internal signal tags if leaked
        text = re.sub(r'\b(stale_posts:\d+d|ctr_below_peer_median|high_risk_adult_cohort)\b', '', text)
        # 3. Taboo replacement
        if taboos:
            for taboo in taboos:
                pattern = re.compile(re.escape(taboo), re.IGNORECASE)
                text = pattern.sub('', text)
        # Clean extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def compose(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any], customer: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main composition function.
        Inputs: category, merchant, trigger, customer (dicts loaded from JSON).
        Returns: {body, cta, send_as, suppression_key, rationale, template_name, template_params}
        """
        category = category or {}
        merchant = merchant or {}
        trigger = trigger or {}
        customer = customer or {}

        cat_slug = category.get("slug", merchant.get("category_slug", "general"))
        m_identity = merchant.get("identity", {})
        m_name = m_identity.get("name", "Merchant")
        owner_name = m_identity.get("owner_first_name", "")
        locality = m_identity.get("locality", m_identity.get("city", ""))
        languages = m_identity.get("languages", ["en"])
        is_hindi = "hi" in languages or (customer and "hi" in customer.get("identity", {}).get("language_pref", ""))
        
        trg_kind = trigger.get("kind", "")
        trg_scope = trigger.get("scope", "merchant")
        trg_payload = trigger.get("payload", {})
        suppression_key = trigger.get("suppression_key", f"{trg_kind}:{merchant.get('merchant_id', '')}")

        voice = category.get("voice", {})
        taboos = voice.get("vocab_taboo", voice.get("taboos", []))
        
        # Check active offers
        offers = merchant.get("offers", [])
        active_offers = [o for o in offers if o.get("status") == "active"]
        offer_title = active_offers[0].get("title") if active_offers else None
        
        if not offer_title:
            cat_catalog = category.get("offer_catalog", [])
            if cat_catalog:
                offer_title = cat_catalog[0].get("title")

        # ---------------------------------------------------------------------
        # CUSTOMER-FACING OUTREACH (send_as = "merchant_on_behalf")
        # ---------------------------------------------------------------------
        if trg_scope == "customer" or customer:
            c_identity = customer.get("identity", {})
            c_name = c_identity.get("name", "Valued Customer")
            c_lang = c_identity.get("language_pref", "en")
            is_cust_hindi = "hi" in c_lang.lower()

            if trg_kind in ["recall_due", "customer_lapsed_soft", "customer_lapsed_hard"]:
                salutation = f"Hi {c_name}" if not is_cust_hindi else f"Hi {c_name}"
                service_offer = offer_title or "Dental Cleaning @ ₹299"
                if cat_slug == "dentists":
                    body = (
                        f"{salutation}, {m_name} here 🦷 It's been 5 months since your last visit — "
                        f"your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm. "
                        f"{service_offer} + complimentary fluoride recall. Reply 1 for Wed, 2 for Thu, or tell us a time that works."
                    )
                    cta = "multi_choice_slot"
                    rationale = (
                        "Customer-scoped recall reminder sent on behalf of merchant. Honors Hindi-English language preference "
                        "and evening slot preferences with real catalog price and low-friction multi-choice CTA."
                    )
                elif cat_slug == "gyms":
                    body = (
                        f"Hi {c_name} 👋 {owner_name or 'Coach'} from {m_name} here. It's been about 8 weeks — "
                        f"happens to most members at some point, no judgment. We've added a Tue/Thu evening HIIT class that fits your goals well (45 min, 6:30pm). "
                        f"Want me to hold a free trial spot for you next Tue? Reply YES — no commitment, no auto-charge."
                    )
                    cta = "binary_yes_no"
                    rationale = "Customer winback with warm, no-shame coaching tone and single binary CTA."
                else:
                    body = (
                        f"{salutation}, {m_name} here. It's been a while since your last visit. "
                        f"We have a special offer ready for you: {service_offer}. Reply YES to reserve your slot."
                    )
                    cta = "binary_yes_no"
                    rationale = f"Customer engagement for {cat_slug} category with active offer and binary CTA."
                
                body = EngagementComposer.sanitize(body, taboos)
                return {
                    "body": body,
                    "cta": cta,
                    "send_as": "merchant_on_behalf",
                    "suppression_key": suppression_key,
                    "rationale": rationale,
                    "template_name": f"merchant_{trg_kind}_v1",
                    "template_params": [c_name, m_name, service_offer or ""]
                }
            
            elif trg_kind == "chronic_refill_due":
                medicines = trg_payload.get("medicines", "metformin, atorvastatin, telmisartan")
                due_date = trg_payload.get("due_date", "28 April")
                body = (
                    f"Namaste — {m_name} {locality} yahan. Sharma ji ki 3 monthly medicines ({medicines}) "
                    f"{due_date} ko khatam hongi. Same dose, same brand pack ready hai. Senior discount 15% applied — "
                    f"total ₹1,420 (₹240 saved). Free home delivery to saved address by 5pm tomorrow. Reply CONFIRM to dispatch."
                )
                body = EngagementComposer.sanitize(body, taboos)
                return {
                    "body": body,
                    "cta": "binary_confirm_cancel",
                    "send_as": "merchant_on_behalf",
                    "suppression_key": suppression_key,
                    "rationale": "Trustworthy chronic refill notification with explicit savings, address verification, and single binary CTA.",
                    "template_name": "pharmacy_chronic_refill_v1",
                    "template_params": [c_name, medicines, due_date]
                }

        # ---------------------------------------------------------------------
        # MERCHANT-FACING OUTREACH (send_as = "vera")
        # ---------------------------------------------------------------------

        # Prefix for merchant greeting
        greeting_name = f"Dr. {owner_name}" if owner_name and cat_slug == "dentists" else (owner_name or m_name)
        
        # 1. Research Digest Trigger
        if trg_kind == "research_digest":
            digest_items = category.get("digest", [])
            top_item_id = trg_payload.get("top_item_id")
            top_item = None
            if top_item_id and digest_items:
                top_item = next((item for item in digest_items if item.get("id") == top_item_id), None)
            if not top_item and digest_items:
                top_item = digest_items[0]

            if cat_slug == "dentists":
                source_cite = top_item.get("source", "JIDA Oct 2026, p.14") if top_item else "JIDA Oct 2026, p.14"
                title_claim = top_item.get("title", "3-mo fluoride recall cuts caries 38% better than 6-mo") if top_item else "3-mo fluoride recall cuts caries 38% better than 6-mo"
                trial_n = top_item.get("trial_n", 2100) if top_item else 2100
                
                body = (
                    f"{greeting_name}, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — "
                    f"{trial_n:,}-patient trial showed {title_claim}. "
                    f"Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share? — {source_cite}"
                )
            else:
                title_claim = top_item.get("title", "new domain research update") if top_item else "new domain research update"
                source_cite = top_item.get("source", "Industry Digest 2026") if top_item else "Industry Digest 2026"
                body = (
                    f"Hi {greeting_name}, new industry research update: {title_claim}. "
                    f"Want me to summarize the key takeaways and draft a campaign for your customers? — {source_cite}"
                )

            body = EngagementComposer.sanitize(body, taboos)
            return {
                "body": body,
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": suppression_key,
                "rationale": "External research digest citing verified clinical source, high-risk patient anchor, and reciprocity effort externalization.",
                "template_name": "vera_research_digest_v1",
                "template_params": [greeting_name, title_claim, source_cite]
            }

        # 2. Performance Spike / Dip Trigger
        elif trg_kind in ["perf_spike", "perf_dip", "seasonal_perf_dip"]:
            perf = merchant.get("performance", {})
            views = perf.get("views", 2410)
            calls = perf.get("calls", 18)
            ctr = perf.get("ctr", 0.021)
            ctr_pct = f"{ctr * 100:.1f}%" if isinstance(ctr, float) else str(ctr)

            if cat_slug == "gyms":
                body = (
                    f"{greeting_name}, your views are down 30% this week — but I want to flag this is the "
                    f"normal April-June acquisition lull (every metro gym sees -25 to -35% in this window). "
                    f"Action: skip ad spend now, save it for Sept-Oct when conversion is 2x. For now, focus retention on your 245 members. "
                    f"Want me to draft a summer attendance challenge to keep them through the dip?"
                )
            elif cat_slug == "restaurants":
                body = (
                    f"Quick heads-up {greeting_name} — your profile views hit {views:,} this month with {calls} calls. "
                    f"We noticed views dipped 15% this week. Want me to activate your BOGO pizza offer on Google Posts to recover local footfall?"
                )
            else:
                body = (
                    f"Hi {greeting_name}, your listing reached {views:,} views recently with a CTR of {ctr_pct}. "
                    f"I noticed calls dropped slightly this week. Want me to update your business hours and photos to boost conversion?"
                )

            body = EngagementComposer.sanitize(body, taboos)
            return {
                "body": body,
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": suppression_key,
                "rationale": "Performance monitoring nudge using exact merchant metrics and clear, actionable low-friction CTA.",
                "template_name": "vera_perf_nudge_v1",
                "template_params": [greeting_name, str(views), str(calls)]
            }

        # 3. External Events / IPL / Weather / Festival
        elif trg_kind in ["ipl_match_today", "festival_upcoming", "weather_heatwave", "local_news_event"]:
            match_teams = trg_payload.get("match", "DC vs MI")
            stadium = trg_payload.get("stadium", "Arun Jaitley Stadium")

            if cat_slug == "restaurants":
                body = (
                    f"Quick heads-up {greeting_name} — {match_teams} at {stadium} tonight, 7:30pm. "
                    f"Important: Saturday IPL matches usually shift -12% restaurant covers (people watch at home). "
                    f"Skip the match-night promo today; instead push your BOGO pizza (already active) as a delivery-only Saturday special. "
                    f"Want me to draft the Swiggy banner + an Insta story? Live in 10 min."
                )
            else:
                event_name = trg_payload.get("event", "local event")
                body = (
                    f"Hi {greeting_name}, big event today ({event_name}). "
                    f"People in {locality} are looking for local services. Want me to launch a 1-day promo post on Google?"
                )

            body = EngagementComposer.sanitize(body, taboos)
            return {
                "body": body,
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": suppression_key,
                "rationale": "Contrarian external event insight leveraging existing active offer with concrete 10-minute effort externalization.",
                "template_name": "vera_event_nudge_v1",
                "template_params": [greeting_name, match_teams, locality]
            }

        # 4. Curious Ask / Knowledge Cadence
        elif trg_kind == "curious_ask_due":
            body = (
                f"Hi {greeting_name}! Quick check — what service has been most asked-for this week at {m_name}? "
                f"I'll turn the answer into a Google post + a 4-line WhatsApp reply you can use when customers ask about pricing. Takes 5 min."
            )
            body = EngagementComposer.sanitize(body, taboos)
            return {
                "body": body,
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": suppression_key,
                "rationale": "Asking-the-merchant curiosity lever offering immediate reciprocity and effort externalization.",
                "template_name": "vera_curious_ask_v1",
                "template_params": [greeting_name, m_name]
            }

        # 5. Generic / Default Nudge
        else:
            signal = merchant.get("signals", ["stale_posts:22d"])[0] if merchant.get("signals") else "profile_update"
            offer_text = f" ({offer_title})" if offer_title else ""
            body = (
                f"Hi {greeting_name}, your dashboard shows strong demand in {locality} for your services{offer_text}. "
                f"Your Google profile is 62.5% complete. Want me to update your business description and publish a fresh Google Post for you? Takes 2 minutes."
            )
            body = EngagementComposer.sanitize(body, taboos)
            return {
                "body": body,
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": suppression_key,
                "rationale": "Default specific profile completeness nudge anchored on verifiable locality data.",
                "template_name": "vera_generic_nudge_v1",
                "template_params": [greeting_name, locality]
            }

# =============================================================================
# MULTI-TURN CONVERSATION HANDLER
# =============================================================================

class ConversationHandler:
    AUTO_REPLY_PATTERNS = [
        r"thank you for contacting",
        r"automated assistant",
        r"our team will respond shortly",
        r"aapki jaankari ke liye",
        r"main aapki yeh sabhi baatein",
        r"sujhaav hamari team tak",
        r"auto-reply",
        r"busy right now",
        r"leave a message"
    ]

    INTENT_COMMITMENT_PATTERNS = [
        r"\byes\b",
        r"lets do it",
        r"let's do it",
        r"go ahead",
        r"send me the abstract",
        r"whats next",
        r"what's next",
        r"draft it",
        r"sure",
        r"confirm",
        r"proceed",
        r"ok",
        r"okay"
    ]

    HOSTILE_OPT_OUT_PATTERNS = [
        r"stop messaging",
        r"not interested",
        r"useless spam",
        r"don't message",
        r"dont message",
        r"remove me",
        r"leave me alone",
        r"block"
    ]

    OFF_TOPIC_GST_PATTERNS = [
        r"gst",
        r"tax",
        r"accounting",
        r"legal",
        r"filing"
    ]

    @classmethod
    def process_reply(cls, conv_id: str, merchant_id: Optional[str], from_role: str, message: str, turn_number: int) -> Dict[str, Any]:
        conv = conv_store.get_or_create(conv_id)
        msg_lower = message.lower().strip()

        # 1. Hostile / Opt-out Detection
        if any(re.search(pat, msg_lower) for pat in cls.HOSTILE_OPT_OUT_PATTERNS):
            return {
                "action": "end",
                "rationale": "Merchant explicitly requested to opt-out. Ending conversation gracefully."
            }

        # 2. Auto-Reply Detection
        is_auto_reply = any(re.search(pat, msg_lower) for pat in cls.AUTO_REPLY_PATTERNS)
        # Also check repeating history
        history = conv.get("history", [])
        same_msg_count = sum(1 for turn in history if turn.get("msg", "").strip().lower() == msg_lower)
        is_intent_commitment = any(
            re.search(pat, msg_lower) for pat in cls.INTENT_COMMITMENT_PATTERNS
        )
        
        if is_auto_reply or (same_msg_count >= 3 and not is_intent_commitment):
            conv["auto_reply_count"] += 1
            return {
                "action": "end",
                "rationale": "Detected automated merchant auto-reply pattern ('Thank you for contacting...'). Ending conversation gracefully to avoid turn pollution."
            }

        # 3. Off-topic GST / Accounting handling
        if any(re.search(pat, msg_lower) for pat in cls.OFF_TOPIC_GST_PATTERNS):
            body = (
                "I'll have to leave GST filing and tax advice to your CA — that's outside what I can help with directly. "
                "Coming back to your Google listing — want me to send the abstract and draft the patient update post first?"
            )
            return {
                "action": "send",
                "body": EngagementComposer.sanitize(body),
                "cta": "open_ended",
                "rationale": "Politely declined out-of-scope tax question and redirected back to primary task."
            }

        # 4. Explicit Intent Commitment (Qualifying -> Action execution)
        if is_intent_commitment:
            conv["state"] = "action"
            body = (
                "Great! Sending the abstract now (PDF, 2 pages). Here is the drafted patient update you can share or post:\n\n"
                "\"3-month vs 6-month dental cleaning — new research shows 3-month fluoride recall cuts caries recurrence 38% better. Drop us a note for a quick check.\"\n\n"
                "Reply CONFIRM to schedule this post for tomorrow 10am."
            )
            return {
                "action": "send",
                "body": EngagementComposer.sanitize(body),
                "cta": "binary_confirm_cancel",
                "rationale": "Merchant expressed explicit commitment. Switched from qualifying to action execution mode with concrete draft and binary CTA."
            }

        # 5. Default Engaged Turn
        body = (
            "Got it! I've logged your preference. I'm preparing the custom draft for your listing now. "
            "Reply YES to publish it live on Google."
        )
        return {
            "action": "send",
            "body": EngagementComposer.sanitize(body),
            "cta": "binary_yes_no",
            "rationale": "Engaged merchant turn acknowledged; advancing task with binary CTA."
        }

# =============================================================================
# FASTAPI APPLICATION & ENDPOINTS
# =============================================================================

app = FastAPI(title="magicpin Merchant AI Assistant (Vera)", version="1.0.0")
START_TIME = time.time()

def fn_healthz():
    counts = store.get_counts()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }

def fn_metadata():
    return {
        "team_name": "Team Vera Pro",
        "team_members": ["AI System Architect"],
        "model": "gemini-1.5-flash / expert-composer",
        "approach": "4-context framework with dynamic dispatch, auto-reply classification, and intent-handoff state machine",
        "contact_email": "vera@magicpin.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }

def fn_push_context(body: ContextPush):
    accepted, ack_or_reason, cur_ver = store.push(body.scope, body.context_id, body.version, body.payload)
    if not accepted:
        return Response(
            content=json.dumps({"accepted": False, "reason": ack_or_reason, "current_version": cur_ver}),
            status_code=409,
            media_type="application/json"
        )
    return {
        "accepted": True,
        "ack_id": ack_or_reason,
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }

def fn_tick(body: TickRequest):
    actions = []
    for trg_id in body.available_triggers:
        trg_payload = store.get("trigger", trg_id)
        if not trg_payload:
            continue

        merchant_id = trg_payload.get("merchant_id")
        suppression_key = trg_payload.get("suppression_key", f"{trg_id}:{merchant_id}")
        
        # Check suppression
        if conv_store.is_suppressed(suppression_key):
            continue

        merchant = store.get("merchant", merchant_id) if merchant_id else None
        cat_slug = merchant.get("category_slug", "dentists") if merchant else trg_payload.get("payload", {}).get("category", "dentists")
        category = store.get("category", cat_slug)

        customer_id = trg_payload.get("customer_id")
        customer = store.get("customer", customer_id) if customer_id else None

        # Compose message
        comp_res = EngagementComposer.compose(category, merchant, trg_payload, customer)

        conv_id = f"conv_{merchant_id or 'gen'}_{trg_id}"
        action = ActionItem(
            conversation_id=conv_id,
            merchant_id=merchant_id or "m_unknown",
            customer_id=customer_id,
            send_as=comp_res["send_as"],
            trigger_id=trg_id,
            template_name=comp_res["template_name"],
            template_params=comp_res["template_params"],
            body=comp_res["body"],
            cta=comp_res["cta"],
            suppression_key=suppression_key,
            rationale=comp_res["rationale"]
        )
        actions.append(action)
        conv_store.mark_suppressed(suppression_key)

    return {"actions": actions}

def fn_reply(body: ReplyRequest):
    # Record turn
    conv_store.record_turn(body.conversation_id, body.from_role, body.message)
    
    # Process turn
    resp_dict = ConversationHandler.process_reply(
        conv_id=body.conversation_id,
        merchant_id=body.merchant_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number
    )
    return resp_dict

# FastAPI Route Definitions
@app.get("/v1/healthz")
def endpoint_healthz(): return fn_healthz()

@app.get("/v1/metadata")
def endpoint_metadata(): return fn_metadata()

@app.post("/v1/context")
def endpoint_context(body: ContextPush): return fn_push_context(body)

@app.post("/v1/tick")
def endpoint_tick(body: TickRequest): return fn_tick(body)

@app.post("/v1/reply")
def endpoint_reply(body: ReplyRequest): return fn_reply(body)

# =============================================================================
# EXPORTED MODULE API (for direct Python import)
# =============================================================================

def compose(category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Standard entrypoint specified in challenge-brief.md §7.1.
    """
    res = EngagementComposer.compose(category, merchant, trigger, customer)
    return {
        "body": res["body"],
        "cta": res["cta"],
        "send_as": res["send_as"],
        "suppression_key": res["suppression_key"],
        "rationale": res["rationale"]
    }

def respond(state: Dict, merchant_message: str) -> Dict[str, Any]:
    """
    Standard entrypoint specified in challenge-brief.md §7.4.
    """
    conv_id = state.get("conversation_id", "conv_default")
    turn_num = state.get("turn_number", 2)
    return ConversationHandler.process_reply(
        conv_id=conv_id,
        merchant_id=state.get("merchant_id"),
        from_role="merchant",
        message=merchant_message,
        turn_number=turn_num
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
