"""Search YouTube per category and keep candidates that carry human-made English subtitles."""
import sys, json
import yt_dlp

QUERIES = {
    "ted_clean": "ytsearch8:TED talk 2023",
    "podcast_conversation": "ytsearch8:podcast interview two hosts full episode clip",
    "technical_lecture": "ytsearch8:MIT OpenCourseWare lecture computer science",
    "news_fast": "ytsearch8:PBS NewsHour full segment",
    "indian_english": "ytsearch8:TED talk India speaker Nilekani OR Indian",
    "noisy_outdoor": "ytsearch8:street interview vlog outdoor noisy subtitles",
}
cats = sys.argv[1:] or list(QUERIES)
opts = {"quiet": True, "skip_download": True, "extract_flat": False, "noplaylist": True, "ignoreerrors": True}
with yt_dlp.YoutubeDL(opts) as ydl:
    for cat in cats:
        res = ydl.extract_info(QUERIES[cat], download=False) or {}
        for e in res.get("entries") or []:
            if not e:
                continue
            manual = [k for k in (e.get("subtitles") or {}) if k.lower().startswith("en")]
            if manual:
                print(json.dumps({"cat": cat, "id": e["id"], "title": e.get("title"), "channel": e.get("channel"),
                                  "dur_s": e.get("duration"), "manual_en": manual, "licence": e.get("license")}), flush=True)
