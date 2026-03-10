"""Dialogue Pipeline Orchestrator for Vietnamese AI Call Center.

Integrates all dialogue components (STT, NLU, DST, Policy, NLG, TTS)
into a unified pipeline for end-to-end conversation handling.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from ..dst.tracker import DialogueState, StateTracker
from ..policy.rule_policy import RuleBasedPolicy, PolicyConfig
from ..nlg.templates import TemplateNLG, NLGConfig

logger = logging.getLogger(__name__)


class NLUProtocol(Protocol):
    """Protocol for NLU models."""

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent and slots from text.

        Args:
            text: Input text.

        Returns:
            Dict with 'intent', 'confidence', and 'slots' keys.
        """
        ...


class STTProtocol(Protocol):
    """Protocol for Speech-to-Text."""

    def transcribe(self, audio: Union[str, Path, bytes]) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio file path or bytes.

        Returns:
            Transcribed text.
        """
        ...


class TTSProtocol(Protocol):
    """Protocol for Text-to-Speech."""

    def synthesize(self, text: str) -> bytes:
        """Convert text to audio.

        Args:
            text: Input text.

        Returns:
            Audio bytes.
        """
        ...


@dataclass
class PipelineConfig:
    """Configuration for dialogue pipeline.

    Attributes:
        confidence_threshold: Minimum NLU confidence
        enable_tts: Whether to generate audio output
        enable_stt: Whether to accept audio input
        max_turns: Maximum conversation turns before reset
        log_turns: Whether to log each turn
    """

    confidence_threshold: float = 0.5
    enable_tts: bool = True
    enable_stt: bool = True
    max_turns: int = 50
    log_turns: bool = True


@dataclass
class TurnResult:
    """Result of processing a dialogue turn.

    Attributes:
        response: Generated text response
        audio_response: Audio bytes (if TTS enabled)
        state: Current dialogue state
        action: Policy action taken
        nlu_output: NLU prediction
        user_input: Original user input
        transcript: STT transcript (if audio input)
    """

    response: str
    audio_response: Optional[bytes] = None
    state: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None
    nlu_output: Optional[Dict[str, Any]] = None
    user_input: str = ""
    transcript: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding audio bytes)."""
        return {
            "response": self.response,
            "has_audio": self.audio_response is not None and len(self.audio_response) > 0,
            "state": self.state,
            "action": self.action,
            "nlu_output": self.nlu_output,
            "user_input": self.user_input,
            "transcript": self.transcript,
        }


class DialoguePipeline:
    """Full dialogue pipeline orchestrating all components.

    Integrates:
    - STT (Speech-to-Text): Convert audio input to text
    - NLU (Natural Language Understanding): Extract intent and slots
    - DST (Dialogue State Tracking): Track conversation state
    - Policy: Decide system action
    - NLG (Natural Language Generation): Generate response text
    - TTS (Text-to-Speech): Convert response to audio

    Usage:
        # Basic setup with NLU model
        from src.nlu import JointBERTNLU
        nlu = JointBERTNLU.load("checkpoints/jointbert")
        pipeline = DialoguePipeline(nlu_model=nlu)

        # Process text input
        result = pipeline.process("Tôi muốn đặt vé đi Đà Nẵng")
        print(result.response)
        print(result.state)

        # With speech
        from src.speech import SpeechService
        speech = SpeechService()
        pipeline = DialoguePipeline(
            nlu_model=nlu,
            stt=speech.stt,
            tts=speech.tts,
        )
        result = pipeline.process_audio(audio_bytes)

        # Reset for new conversation
        pipeline.reset()
    """

    def __init__(
        self,
        nlu_model: Optional[NLUProtocol] = None,
        tracker: Optional[StateTracker] = None,
        policy: Optional[RuleBasedPolicy] = None,
        nlg: Optional[TemplateNLG] = None,
        stt: Optional[STTProtocol] = None,
        tts: Optional[TTSProtocol] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize dialogue pipeline.

        Args:
            nlu_model: NLU model for intent/slot prediction.
            tracker: Dialogue state tracker.
            policy: Policy for action selection.
            nlg: NLG for response generation.
            stt: Speech-to-text component.
            tts: Text-to-speech component.
            config: Pipeline configuration.
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.nlu = nlu_model
        self.tracker = tracker or StateTracker()
        self.policy = policy or RuleBasedPolicy()
        self.nlg = nlg or TemplateNLG()
        self.stt = stt
        self.tts = tts

        # Conversation history
        self._history: List[TurnResult] = []

        logger.info(
            "DialoguePipeline initialized: NLU=%s, STT=%s, TTS=%s",
            type(self.nlu).__name__ if self.nlu else "None",
            type(self.stt).__name__ if self.stt else "None",
            type(self.tts).__name__ if self.tts else "None",
        )

    def reset(self) -> DialogueState:
        """Reset pipeline for new conversation.

        Returns:
            Fresh dialogue state.
        """
        self.tracker.reset()
        self.policy.reset()
        self._history = []
        logger.info("Pipeline reset for new conversation")
        return self.tracker.state

    def process(self, user_input: str) -> TurnResult:
        """Process a text input turn.

        Args:
            user_input: User's text input.

        Returns:
            TurnResult with response and updated state.
        """
        # Check for max turns
        if self.tracker.state.turn_count >= self.config.max_turns:
            logger.warning("Max turns reached, resetting conversation")
            self.reset()

        # Run NLU
        nlu_output = self._run_nlu(user_input)

        # Update dialogue state
        state = self.tracker.update(nlu_output, user_text=user_input)

        # Select action
        action = self.policy.select_action(state, nlu_output)
        self.tracker.set_last_action(action)

        # Generate response
        response = self.nlg.generate(action, state.to_dict())

        # Generate audio if TTS enabled
        audio_response = None
        if self.config.enable_tts and self.tts:
            try:
                audio_response = self.tts.synthesize(response)
            except Exception as e:
                logger.warning("TTS failed: %s", e)

        # Create result
        result = TurnResult(
            response=response,
            audio_response=audio_response,
            state=state.to_dict(),
            action=action,
            nlu_output=nlu_output,
            user_input=user_input,
        )

        # Store in history
        self._history.append(result)

        # Log turn
        if self.config.log_turns:
            logger.info(
                "Turn %d: input='%s' intent=%s conf=%.2f action=%s",
                state.turn_count,
                user_input[:50],
                nlu_output.get("intent"),
                nlu_output.get("confidence", 0),
                action.get("type"),
            )

        return result

    def process_audio(self, audio: Union[str, Path, bytes]) -> TurnResult:
        """Process an audio input turn.

        Args:
            audio: Audio file path or bytes.

        Returns:
            TurnResult with response, audio, and transcript.

        Raises:
            RuntimeError: If STT is not configured.
        """
        if not self.stt:
            raise RuntimeError("STT not configured. Initialize pipeline with stt parameter.")

        # Transcribe audio
        try:
            transcript = self.stt.transcribe(audio)
        except Exception as e:
            logger.error("STT failed: %s", e)
            # Return error response
            return TurnResult(
                response="Xin lỗi, em không nghe rõ. Anh/chị có thể nói lại được không ạ?",
                user_input="[audio]",
                transcript="[transcription failed]",
            )

        if not transcript:
            return TurnResult(
                response="Xin lỗi, em không nghe thấy gì. Anh/chị có thể nói lại được không ạ?",
                user_input="[audio]",
                transcript="",
            )

        # Process transcribed text
        result = self.process(transcript)
        result.transcript = transcript

        return result

    def _run_nlu(self, text: str) -> Dict[str, Any]:
        """Run NLU prediction.

        Args:
            text: Input text.

        Returns:
            NLU output dict.
        """
        if not self.nlu:
            # Return mock NLU output when no model is configured
            logger.warning("NLU not configured, returning mock output")
            return self._mock_nlu(text)

        try:
            return self.nlu.predict(text)
        except Exception as e:
            logger.error("NLU prediction failed: %s", e)
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "slots": {},
            }

    def _mock_nlu(self, text: str) -> Dict[str, Any]:
        """Generate mock NLU output for testing.

        Args:
            text: Input text.

        Returns:
            Mock NLU output.
        """
        text_lower = text.lower()

        # Simple keyword-based intent detection
        intent = "unknown"
        confidence = 0.5
        slots: Dict[str, str] = {}

        # Flight booking keywords
        if any(w in text_lower for w in ["đặt vé", "dat ve", "book", "bay", "chuyến bay"]):
            intent = "flight"
            confidence = 0.8

        # Airfare keywords
        elif any(w in text_lower for w in ["giá", "gia", "bao nhiêu", "fare", "price"]):
            intent = "airfare"
            confidence = 0.75

        # Greeting keywords
        elif any(w in text_lower for w in ["xin chào", "chao", "hello", "hi"]):
            intent = "greet"
            confidence = 0.9

        # Extract city mentions
        city_keywords = {
            "hà nội": "ha noi",
            "ha noi": "ha noi",
            "sài gòn": "ho chi minh",
            "sai gon": "ho chi minh",
            "hồ chí minh": "ho chi minh",
            "ho chi minh": "ho chi minh",
            "đà nẵng": "da nang",
            "da nang": "da nang",
            "nha trang": "nha trang",
            "huế": "hue",
            "hue": "hue",
            "phú quốc": "phu quoc",
            "phu quoc": "phu quoc",
        }

        # Find all city mentions with their positions
        city_mentions = []
        for city_vi, city_en in city_keywords.items():
            pos = text_lower.find(city_vi)
            if pos != -1:
                city_mentions.append((pos, city_vi, city_en))

        # Sort by position and process
        city_mentions.sort(key=lambda x: x[0])

        for pos, city_vi, city_en in city_mentions:
            # Check context before city name
            before = text_lower[:pos]
            after_pos = pos + len(city_vi)

            # Check for "từ" (from) - look for closest preposition
            from_indicators = ["từ ", "tu "]
            to_indicators = ["đến ", "den ", "tới ", "toi ", "di "]

            # Find the closest preposition before this city
            last_from_pos = -1
            last_to_pos = -1

            for indicator in from_indicators:
                idx = before.rfind(indicator)
                if idx > last_from_pos:
                    last_from_pos = idx

            for indicator in to_indicators:
                idx = before.rfind(indicator)
                if idx > last_to_pos:
                    last_to_pos = idx

            # Determine slot based on closest preposition
            if last_from_pos > last_to_pos:
                if "fromloc.city_name" not in slots:
                    slots["fromloc.city_name"] = city_en
            elif last_to_pos > last_from_pos:
                if "toloc.city_name" not in slots:
                    slots["toloc.city_name"] = city_en
            else:
                # No preposition found - use as toloc by default if not set
                if "toloc.city_name" not in slots:
                    slots["toloc.city_name"] = city_en
                elif "fromloc.city_name" not in slots:
                    slots["fromloc.city_name"] = city_en

        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots,
        }

    @property
    def history(self) -> List[TurnResult]:
        """Get conversation history."""
        return self._history

    @property
    def state(self) -> DialogueState:
        """Get current dialogue state."""
        return self.tracker.state

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation.

        Returns:
            Summary dictionary.
        """
        return {
            "turn_count": self.tracker.state.turn_count,
            "current_intent": self.tracker.state.current_intent,
            "slots": self.tracker.state.slots,
            "completed": self.tracker.state.completed,
            "confirmed": self.tracker.state.confirmed,
            "history_length": len(self._history),
        }


def create_pipeline(
    nlu_model: Optional[NLUProtocol] = None,
    speech_service: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> DialoguePipeline:
    """Factory function to create a dialogue pipeline.

    Args:
        nlu_model: NLU model instance.
        speech_service: SpeechService instance (provides both STT and TTS).
        config: Configuration dictionary.

    Returns:
        Configured DialoguePipeline instance.
    """
    pipeline_config = None
    if config:
        pipeline_config = PipelineConfig(
            confidence_threshold=config.get("confidence_threshold", 0.5),
            enable_tts=config.get("enable_tts", True),
            enable_stt=config.get("enable_stt", True),
            max_turns=config.get("max_turns", 50),
            log_turns=config.get("log_turns", True),
        )

    stt = None
    tts = None
    if speech_service:
        stt = speech_service.stt
        tts = speech_service.tts

    return DialoguePipeline(
        nlu_model=nlu_model,
        stt=stt,
        tts=tts,
        config=pipeline_config,
    )


class ConversationManager:
    """Manages multiple conversations with session support.

    Useful for multi-user scenarios or web applications where
    each user has their own conversation state.

    Usage:
        manager = ConversationManager(nlu_model=nlu)

        # Create session for user
        session_id = manager.create_session()

        # Process user input
        result = manager.process(session_id, "Đặt vé đi Đà Nẵng")

        # Get session state
        state = manager.get_state(session_id)

        # End session
        manager.end_session(session_id)
    """

    def __init__(
        self,
        nlu_model: Optional[NLUProtocol] = None,
        speech_service: Optional[Any] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize conversation manager.

        Args:
            nlu_model: NLU model to use for all sessions.
            speech_service: Speech service for STT/TTS.
            config: Pipeline configuration.
        """
        self.nlu_model = nlu_model
        self.speech_service = speech_service
        self.config = config
        self._sessions: Dict[str, DialoguePipeline] = {}
        self._session_counter = 0

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session.

        Args:
            session_id: Optional custom session ID.

        Returns:
            Session ID.
        """
        if session_id is None:
            self._session_counter += 1
            session_id = f"session_{self._session_counter}"

        stt = self.speech_service.stt if self.speech_service else None
        tts = self.speech_service.tts if self.speech_service else None

        self._sessions[session_id] = DialoguePipeline(
            nlu_model=self.nlu_model,
            stt=stt,
            tts=tts,
            config=self.config,
        )

        logger.info("Created session: %s", session_id)
        return session_id

    def process(self, session_id: str, user_input: str) -> TurnResult:
        """Process input for a session.

        Args:
            session_id: Session identifier.
            user_input: User's text input.

        Returns:
            Turn result.

        Raises:
            KeyError: If session doesn't exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id].process(user_input)

    def process_audio(self, session_id: str, audio: Union[str, Path, bytes]) -> TurnResult:
        """Process audio input for a session.

        Args:
            session_id: Session identifier.
            audio: Audio data.

        Returns:
            Turn result.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id].process_audio(audio)

    def get_state(self, session_id: str) -> DialogueState:
        """Get state for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Dialogue state.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id].state

    def end_session(self, session_id: str) -> None:
        """End and cleanup a session.

        Args:
            session_id: Session identifier.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Ended session: %s", session_id)

    def reset_session(self, session_id: str) -> DialogueState:
        """Reset a session for new conversation.

        Args:
            session_id: Session identifier.

        Returns:
            Fresh dialogue state.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id].reset()

    @property
    def active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self._sessions.keys())
