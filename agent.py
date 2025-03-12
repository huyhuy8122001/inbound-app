import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector, elevenlabs
from livekit.plugins.elevenlabs import tts


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            """
            Bạn là trợ lý giọng nói của ePacific Telecom. Bạn giao tiếp với khách hàng qua giọng nói và cần đưa ra câu trả lời ngắn gọn, rõ ràng, tránh sử dụng dấu câu khó phát âm. Bạn được tạo ra để giới thiệu khả năng của các giải pháp CCALL, Eone, AI Agent do ePacific Telecom cung cấp.
            """
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # This project is configured to use Deepgram STT, OpenAI LLM and Cartesia TTS plugins
    # Other great providers exist like Cerebras, ElevenLabs, Groq, Play.ht, Rime, and more
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.stt.STT(
            model="nova-2",
            # model="whisper-medium",
            interim_results=True,
            smart_format=True,
            punctuate=True,
            filler_words=True,
            profanity_filter=False,
            # keywords=[("LiveKit", 1.5)],
            language="vi",
        ),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.tts.TTS(
            model="eleven_turbo_v2_5",
            voice=elevenlabs.tts.Voice(
            id="mvYbQ2cRw9pAg9c9WOAc",
            name="Vietlike",
            # id="bWvs6tS24bngxwBo8QJy",
            # name="Tố Uyên",
            #### noted
            # id="HAAKLJlaJeGl18MKHYeg",
            # name="Trang",
            category="premade",
            settings=elevenlabs.tts.VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
                ),
            ),
            language="vi",
            # streaming_latency=3,
            # enable_ssml_parsing=False,
            # chunk_length_schedule=[80, 120, 200, 260],
        ),
        turn_detector=turn_detector.EOUModel(),
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Xin chào quý khách, tôi là nhân viên số bên ePacific Telecom. Tôi có thể giúp gì cho bạn ngày hôm nay?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # giving this agent a name of: "inbound-agent"
            agent_name="inbound-agent-test",
        ),
    )
