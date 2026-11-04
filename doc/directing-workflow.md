# Directing a production with MiniMax-H3 Studio

This guide is the working procedure for an agent directing a multi-shot film, animation, documentary,
presenter video, commercial or MV through the Studio HTTP API and ffmpeg. The agent develops the story,
characters, dialogue, narration, locations, reference images, storyboard, prompts, jobs, edit and subtitles.
The deliverable is the complete assembled film, with reviewed source-language and translated subtitles
embedded as two selectable tracks. A collection of clips is not a finished production.

## 1. Confirm the brief before production

Ask for the following together, reuse answers already given, and record them in `00_project.md`:

| Decision | What to confirm |
| --- | --- |
| Subject and audience | Story/topic, purpose, intended viewer, desired length, required content and constraints |
| Frame | Landscape or portrait; recommend 16:9 for a film and 9:16 for mobile delivery |
| Resolution | **768p by default**; offer 544p when the user prioritizes speed |
| Video mode | **Reference to video**, using fixed character and location images |
| Spoken language | Confirm the default language of dialogue and narration; recommend Mandarin Chinese if appropriate |
| Visual treatment | Anime/2D, stylized 3D, or realistic/live-action; recommend a treatment suited to the brief |
| Subtitle languages | Source-language subtitles plus the user's requested translation language; ask which track should be default |

Recommend a specific visual direction with a reason rather than offering only a list of styles. For example,
a folklore adventure may suit stylized animation and a present-day suspense story may suit cinematic realism.
Do not silently choose the spoken language or the translation language. Do not start image/video generation
until the required settings are settled. No GPU jobs are needed merely to explain or update this guide.

## 2. Create the production folder and write the whole story first

Create a new folder under `outputs/<title>/` for this production; do not overwrite another production.
Use a meaningful title, with a suffix if necessary. All generated files and supporting records stay here.

```text
outputs/<title>/
  00_project.md             brief, confirmed settings, premise, complete plot, style and delivery plan
  01_bible.md               characters, voices, dialogue, narration, locations and continuity rules
  02_storyboard.md          complete numbered shot list and transitions
  03_reference_manifest.json  selected reference files, versions, roles, job IDs and seeds
  04_generation_log.md      takes, QC, continuity problems and fixes
  PROGRESS.md               current state and exact next steps for another session
  refs/characters/          fixed character images and explicit costume/age variants
  refs/locations/           fixed empty sets and consistent views of them
  refs/shots/               optional compositions derived from the fixed references
  refs/candidates/          unselected image candidates
  prompts/images/           exact reference-image generation/edit prompts
  prompts/video/            exact H3 prompts, one file per segment
  clips/                    accepted segments; rejected/ keeps replaced takes
  qc/                       contact sheets and continuity checks
  scripts/                  prompt builder, API client, assembly and subtitle helpers
  jobs.json                 image/video/ASR requests, asset order, IDs, seeds and outputs
  subs/                     raw ASR, reviewed source SRT, translation SRT and correction log
  previews/                 optional assembly checks
  <title>.clean.mp4         final picture/audio edit before subtitle muxing
  <title>.mp4               delivered film with two embedded subtitle tracks
  cover.png                 title poster; first frame of clips/S00_opening.mp4
  publish.md
```

Keep committed documentation and reusable scripts in English. Production notes may use the user's language;
H3 prompts use English descriptions with dialogue in its actual spoken language. Every local Python command
must use `/Users/zxc/miniconda3/bin/python` (conda base), including helper scripts and pip.

Before storyboarding or generating reference images, write down:

1. **Story:** logline, setting, stakes, complete beginning/middle/end, emotional arc and approximate runtime.
2. **Opening:** an arresting image, event, question or contradiction in the first 5–15 seconds. Introduce enough
   context to understand the stakes without a long exposition dump. Deliver on the opening's promise later.
3. **Characters:** names, motivation, relationships, age, build, face/hair, costume, distinctive marks and props.
4. **Dialogue:** exact words, speaker, intention, emotional delivery and story function. Keep voices distinguishable.
5. **Narration:** whether it exists, narrator identity, perspective, fixed voice and exact lines. Avoid explaining
   everything already visible; use narration for context, tension or information the image cannot convey.
6. **Locations:** architecture, floor plan, landmark positions, entrances, materials, palette, lighting, time,
   weather and recurring sound. Track intentional changes to a location's state.
7. **Film language:** a fixed palette, contrast, lens/framing tendencies, camera behavior, texture and sound strategy.

Work with the deliberate composition, color dramaturgy and spatial storytelling associated with Zhang Yimou,
and the clear action geography, physical cause and effect and large-scale reveals associated with James Cameron.
Turn these references into concrete direction appropriate to this story. A director's name alone is insufficient.
Use an intentional visual grammar throughout rather than changing style with every scene.

Write the full plan first, then proceed through storyboard, reference production, video generation and delivery.
There is no mandatory two-minute approval gate. Honor any review checkpoints the user actually requests;
otherwise continue through the authorized production and keep the user informed.

## 3. Build the storyboard and allocate time

Divide the complete story into **5–15 second segments**, each generated as one continuous shot.
Choose durations for the content; do not force every segment to 15 seconds. Repeated uses of `[Shot 1]` in
separate jobs are correct: each job starts its own local timeline.

Each storyboard entry must contain:

- Segment ID, requested seconds, expected frames, actual seconds and position in the final edit.
- Story purpose and new information; reference IDs for cast, location and any essential prop.
- Opening state: character placement, gaze, pose, prop ownership, movement, light and camera orientation.
- Composition: shot size, viewpoint, lens feel, foreground/midground/background, focus and motivated camera move.
- Timed action: start, development, consequence and reaction, with precise physical and visual details.
- Exact dialogue/narration, speaker IDs, voice descriptions, delivery and allocated speaking windows.
- Synchronous effects, ambience and any score description.
- Closing state and the next segment's opening; cut motivation and continuity requirements.

### Speech and action budget

These are conservative **planning heuristics**, not model guarantees. For Mandarin, start at about
3 Chinese characters per spoken second; allow extra time for emotion, turns, breaths and action.
For English, start around 2–2.5 words per spoken second, then adjust for the voice and complexity.
Calculate `speech time + pauses + non-overlapping action + opening/ending handles <= actual segment duration`.
Dialogue and narration share the budget. Do not count overlapping speech as extra capacity.

| Requested length | Initial Mandarin budget, total across all voices | Practical action plan |
| --- | --- | --- |
| 5 s | About 8–12 characters; normally one brief line | One clear action or reaction |
| 10 s | About 20–25 characters; one or two short lines | One main action with a readable response |
| 15 s | About 30–40 characters; two or three short turns | Two or three connected beats, with a visible outcome |

An action-heavy segment needs less speech. A quiet monologue can use more of the timeline, but avoid
filling its final fraction of a second with words. Reserve roughly 0.5–1 second at boundaries where useful.
If a segment is overloaded, shorten its words or split it; do not solve it by demanding rushed delivery.
Describe rich texture, micro-expressions, material response and light without inventing too many actions
for the allotted seconds. One camera path and clear causal progression are usually easier to read.

The server snaps frames to a 17k+5 grid at 24 fps. Use `frame_count()` from `server/model_catalog.py` in
helpers, or its equivalent below, and plan to the actual duration rather than the requested integer:

```python
def frames(seconds):
    n = max(5, round(seconds * 24))
    return n + (5 - n % 17) % 17
```

Examples: 5 s -> 124 frames (5.1667 s), 10 s -> 243 (10.125 s), 12 s -> 294 (12.25 s),
15 s -> 362 (15.0833 s). Verify actual output frames before final assembly and subtitle review.

### Design the joins before rendering

Keep a continuity ledger for each neighboring pair: screen direction, eyelines, axis, pose, speed,
prop state, geography, lighting and sound. Write how the outgoing state leads into the incoming state.
Use motivated reaction cuts, matched motion, a reveal/payoff, or an establishing shot for a location change.
Maintain the 180-degree axis unless the change is deliberately established. Avoid repeating the same action
on both sides of a cut or teleporting a prop between hands.

Separate jobs do not share audio buffers or motion state. A prompt cannot guarantee a seamless voice or music
bridge between them. Finish a sentence within its segment; place cuts at natural pauses or sound events.
Use the same ambience and score specification, but review the assembled result for restarts and tonal jumps.
If necessary, simplify the music or choose a quiet cut rather than claiming the score will be sample-continuous.

## 4. Generate and freeze character and location references

**Lock character appearance first. Lock each location before rendering a video set there.**
Generate references with Studio's current picture models; a casting video is optional, not a prerequisite.
This applies to both short stories and novel chapters. Principal and recurring characters need independent
reference cards; one-off supporting characters, passersby and background crowds may be described directly
in video prompts without separate cards. Register a stable reference if an incidental character recurs later.

For a novel, search and reuse the shared library under `outputs/<novel-title>/refs/` before generating assets;
all selected character and location images must remain within that novel folder, accessible across chapters.
For a standalone short story, keep the same necessary library within its own production folder.
The reference manifest must be searchable by stable identity/location ID, name/alias, version, traits/state,
chapter/scene applicability and local image path, alongside the generation records. Record the exact selected
versions in each segment's job manifest.

Read `GET /api/presets` for available image choices. At the time of writing:

| Model key | Reference-production use |
| --- | --- |
| `moss` | Text-to-image character/set concepts and posters; no image editing |
| `flux-klein-4b` | Text-to-image and image editing for derived views or compositions |
| `qwen-image-21-turbo` | Text-to-image and image editing; edits require `aspect_ratio=image` |

Pick one coherent rendering style for both people and environments. Produce a small candidate set, inspect it,
and select a canonical image for each required character card and location. Store the file, prompt, model, job ID, seed,
version and intended role in the manifest. A repeated seed alone does not establish identity.

**Character references:** one clearly visible person per selected image, readable face, proportions,
hair, costume and distinctive marks; neutral expression, unobstructed hands where relevant, plain background.
Prefer a clear full/three-quarter view; create a closer face view only when needed. Derive new views and
costume variants from the selected canonical image using image editing rather than independently reinventing
the character. Check identity after every edit. For characters with cards, growth, aging and any change of
clothing/accessories require a new reference version before the affected video is generated, even for a brief
costume change. Record the applicable chapter/scene range, preserve the old image and never overwrite a
version used in a finished video. One-off extras remain covered by the prompt-only exception above.

**Location references:** an empty set without incidental people, readable layout and landmarks, stable
materials/colors, agreed lighting and weather. Write a simple spatial plan. Derive reverse angles and details
from the canonical view, then check that doors, windows, stairs and props occupy compatible positions.
Reuse these images across chapters; register derived versions for changed lighting, weather or set state.
Empty sets prevent background people from competing with the intended cast references.

**Shot compositions:** when placement is complex, derive a single composition from selected character and
location images using an editing model (currently at most four inputs). Record what it controls and verify
that it has not changed faces, costume or geometry. It remains a reference, not a guaranteed first frame.

Use only necessary references in a video job: normally one or two characters plus one location; the API
accepts at most six images. Even establishing shots carry their location reference. Freeze selected assets
before bulk video submission. Do not silently replace a canonical reference halfway through the production.

Example text-to-image request (use the confirmed aspect ratio):

```bash
curl -fsS -X POST "$STUDIO_URL/api/jobs" \
  -F kind=image -F image_model=flux-klein-4b \
  -F 'prompt=<prompts/images/location_workshop.txt' -F aspect_ratio=16:9
```

For a derived view with Qwen, upload the canonical reference and keep its canvas:

```bash
curl -fsS -X POST "$STUDIO_URL/api/jobs" \
  -F kind=image -F image_model=qwen-image-21-turbo \
  -F 'prompt=<prompts/images/workshop_reverse.txt' -F aspect_ratio=image \
  -F ref_image=@refs/locations/workshop.png
```

Poll until complete and download `picture_url`. Reference prompts must describe the actual character or
set in full, including the production's visual treatment. Do not send storyboard placeholders to the API.

## 5. Fix voices in the bible and repeat them in video prompts

For every speaking character and narrator, define a stable speaker ID such as `(S1)`, fixed English voice
description and a user-readable description. Specify age/gender where relevant, pitch, timbre, resonance,
pace, articulation, accent/language and habitual delivery. Keep each voice clearly distinguishable.
The narrator is a separate speaker even when never visible.

Repeat the same voice string at every relevant vocal event, followed by the exact words to say. Emotion may
change locally; identity should not. Generate dialogue and narration with H3 inside the video job; do not
substitute external TTS or separately generated speech. Voice wording encourages consistency but is not a
speaker embedding or a guaranteed voice lock. Review drift and revise problematic takes if needed.

## 6. H3 prompt design: self-contained official format and project template

The format rules below summarize MiniMax's official
[full-reference guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/.agents/skills/h3-prompt-writing/references/ref-en.txt)
and [base audiovisual guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/.agents/skills/h3-prompt-writing/references/base-en.txt),
checked on 2026-10-28. They are embedded here so routine production does not require another search.
The timing budgets, reference-lock procedure and continuity ledger above are Studio production conventions.

### Official format essentials

Use six English sections, in this order:
`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
`overall_soundscape`, `non_diegetic_music`.

Define each referenced entity and its role. `<Subject N>` names reusable visible content, including a location;
`<Picture N>` identifies an uploaded image. Keep meanings consistent. Use `[reference generation]` for this task.
Retention markers include `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference`;
choose according to the defined role. A person reference can preserve identity without preserving its backdrop.

Introduce style before `[Shot 1]`. The first shot has no timestamp; later shot markers denote actual cuts,
not action phases. Describe composition, environment, light, action, state changes, camera and synchronized sound.

Give voices stable `(S1)` IDs. Write camera movement naturally, with scale/speed where useful.
Keep delivery instructions outside `<d>`; put the language and exact spoken words inside:
`<d>[Chinese] ...</d>`. For voiceover use `says in an off-screen voiceover`; keep visible non-speakers' lips closed.
Within one job, `<scenetrans>` denotes speech spanning a cut; `<cutoff>` denotes speech truncated at the end.
Neither creates continuity between separate API jobs.

`overall_soundscape` covers ambience, physical effects and nonverbal sounds; do not repeat dialogue there.
`non_diegetic_music` describes instrumentation, tempo and dynamics, or `N/A` for no background score.

### Studio implementation rules

Build prompts from a script and the frozen bible; do not paraphrase identities/voices independently per job.
Project asset IDs stay global, but `<Picture N>` always follows the **actual upload order for that request**.
Maintain an explicit mapping from project assets to prompt labels in `jobs.json`. Speaker IDs remain stable
throughout the film. The prompt builder must resolve every placeholder before submission.

Each segment has one `[Shot 1]`; express its action phases in prose with local elapsed times.
Do not reuse the old `[Shot 1, 00:00-00:05]` syntax. State actual segment duration in `summary`.
Make every important action observable: which hand, which object, where it moves, what resists, what changes,
and how the character reacts. Establish spatial anchors before moving the camera. Specify gaze, breathing,
fabric, weather, reflections and focus changes when they contribute to the moment. Avoid contradictory
camera commands and details that cannot be seen at the chosen framing.

Keep shots free of readable text, subtitles, logos and watermarks unless the brief requires otherwise.
Subtitle delivery happens after assembly. Reference images normally supply appearance/layout rather than
exact first/last frames. Do not describe a neutral character portrait as the location of the story.
In this server's Ref2VA route use `ref_image`, never `image` or `last_image`; the server rejects mixing them.
An extracted accepted frame may be an additional continuity reference, but does not enforce a first frame.

### Complete original example: one character plus one location

This illustrative segment assumes Mandarin, 16:9, a 10 s request (10.125 s actual), character image first
and empty location image second. Its spoken line, action budget and final gaze set up a reaction cut.
Replace the sample cast/set with the project's canonical descriptions; do not add unrelated image references.

```text
subject_definitions:
<Subject 1> is Lin, a woman in her early thirties with a narrow face, straight black chin-length hair, a small notch in her left eyebrow, a charcoal cotton jacket and a faded rust scarf, shown in <Picture 1>. This image supplies her identity and costume only.
<Subject 2> is the empty mountain radio workshop shown in <Picture 2>, with a north-facing rain-streaked window on screen left, a scarred oak workbench beneath it, an unlit brass desk lamp, and a green steel door on screen right. This image supplies the room layout, materials and lighting.

summary:
[reference generation] A continuous 10.125-second shot follows <Subject 1> finding an impossible signal inside <Subject 2>. Her attention shifts from the radio to the closed door.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - Keep Lin's facial proportions, eyebrow notch, haircut, jacket and rust scarf; the portrait background is not part of her reference role.
<Subject 2> (appears in [Shot 1]): fully_preserved - Keep the workshop's window, workbench, lamp and door in their established spatial relationship, with the same wet surfaces and cool daylight.

detailed_description:
Cinematic realism with restrained slate blue and rust colors, soft overcast window light, natural skin texture and gentle highlight roll-off. No readable text, subtitles, captions, logos or watermarks.
[Shot 1] A medium close view places <Subject 1> in the left half of the frame at the workbench in <Subject 2>. The green door remains visible on the right, separated from her by negative space. Her left hand rests beside the radio; her right thumb and forefinger hold its tuning dial. Her scarf is damp against the collar. A slow, short push toward her face preserves the door in the background and keeps the camera on the room's established side of the axis.
During the first two seconds she rotates the dial a few millimeters. The worn knob offers slight resistance; the radio casing stays planted beneath her left palm. A thin electrical whistle breaks through the static. Her fingers stop immediately, while one bead of rainwater slips from her sleeve onto the oak surface.
Between two and four seconds she leans closer without changing her grip. Her lower eyelids tighten; her mouth stays closed as she listens. Focus eases from the tuning fingers to her eyes, while the cool window reflection remains visible along her cheek. The door stays softly legible behind her.
Between four and six seconds, <Subject 1>, in the low, lightly husky voice of an adult woman, speaking Mandarin Chinese with deliberate articulation and an unhurried pace (S1), says quietly: <d>[Chinese] 这不是求救，是警告。</d> Her delivery is controlled, with a short pause at the comma; her lips follow this line only. The dial remains still throughout the sentence.
From six seconds onward, a single dull knock comes from the door on screen right. She closes her mouth, holds her breath, then turns only her head and eyes toward it. Her shoulders remain angled toward the bench and both hands retain their positions. The camera slows to rest. In the final second she holds the rightward gaze, leaving the door shut and space on screen right for the following segment's view of the threshold. There is no internal cut, no extra speech and no change to another room.

overall_soundscape:
Fine rain taps the window above a low room tone. Radio static briefly narrows into a whistle, a droplet touches wood, and one muted knock sounds at the green door. A faint breath stops after the knock.

non_diegetic_music:
N/A
```

For a narrator, use the fixed voice at the timed narration event, for example:

```text
The unseen narrator, in the warm, low-register voice of a middle-aged man, speaking Mandarin Chinese with measured articulation and a steady pace (S2), says in an off-screen voiceover: <d>[Chinese] 门外的人，已经等了三十年。</d> Lin's lips remain closed throughout the narration.
```

Do not add this example narration to the sample segment without recalculating its time budget.

## 7. Submit video jobs through the API

Use `lora=fasth3-omniref` for every segment, `resolution=768p` unless confirmed otherwise, the confirmed
aspect ratio and the frozen references. Do not send sampler/steps overrides: this model uses its fixed setup.
Set `STUDIO_URL` from the project's deployment configuration; do not commit host secrets into production docs.

```bash
curl -fsS -X POST "$STUDIO_URL/api/jobs" \
  -F 'prompt=<prompts/video/S001.txt' \
  -F resolution=768p -F lora=fasth3-omniref -F aspect_ratio=16:9 -F duration=10 \
  -F ref_image=@refs/characters/lin.png \
  -F ref_image=@refs/locations/workshop.png
```

This makes `<Picture 1>` Lin and `<Picture 2>` the workshop. Save the request, ordered reference mapping,
job ID, returned seed, prompt path, duration and later output path in `jobs.json` immediately after submission.
The image list must match the prompt; do not rely on filenames to convey reference roles to H3.

Multiple ready segments may be submitted together: each is a separate `POST`, and the server queues them
FIFO on one GPU worker. Finish reference dependencies first. A segment that needs a frame from an earlier
result must wait for that result; independent segments can be queued immediately. Never duplicate jobs just
because a polling request timed out; inspect existing IDs first.

Poll `GET /api/jobs/{id}` about every 15 seconds in a background helper, download `video_url` on `done`,
and log `failed`/cancelled results. Keep updates to the user going during long waits. Generate in manageable
submission groups if useful, without imposing a mandatory user approval between groups.

## 8. Review segments and their joins

Create contact sheets (including beginning/end frames) and check audio presence/duration. For example:

```bash
ffmpeg -i clips/S001.mp4 -vf 'fps=1/2,scale=320:-1,tile=8x1' -frames:v 1 qc/S001.jpg
ffmpeg -i clips/S001.mp4 -af volumedetect -vn -f null -
```

Check identity/costume, location geometry/light, intended action and reaction, prop states, anatomy,
unwanted text, timing and opening/closing state. Review adjacent clips together, not only in isolation.
Inspect speech completeness later against the ASR result as well. Text and frame inspection cannot fully
judge pronunciation or exact voice timbre; describe the verification actually performed and invite listening
feedback where needed, without abandoning the remaining production work.

Redo missing story actions, wrong/duplicated characters, broken anatomy, obvious set discontinuities,
truncated essential speech or unwanted burned-in subtitles. Tighten the direction or timing when the problem
is structural; changing a seed alone may not fix it. Keep rejected takes and log the replacement.
Do not cancel a healthy running GPU job just to speed iteration. Fix a recurring issue in the bible/builder
before generating more affected shots. Preserve acceptable minor variations rather than chasing endless takes.

## 9. Make the title cover and animated opening

Every film opens with a 3–5 second animated title shot, like the opening card of a feature film. Make it after
the references and the style are frozen, and before assembly.

**Cover picture.** Write `prompts/images/cover.txt` and generate `cover.png` with Studio's picture API
(`image_model=qwen-image-21-turbo`). Use edit mode with the frozen references as repeatable `ref_image` inputs
so the principal character(s), costume, set and style match the film; put the location (or another image with
the film's aspect ratio) first, because Qwen edits follow the first input's canvas (`aspect_ratio=image`).
Design it as a poster that stops the scroll:

- one strong, simple composition: the protagonist in a characteristic pose in the key location, with a clear
  focal point and room for the title (rule of thirds, not a centered passport photo);
- strong light and color dramaturgy, for example a single warm pool of light against a cold surrounding;
- the film title as large, legible lettering in a style that suits the story (brush calligraphy, storybook
  type, bold poster lettering); spell out the exact title in quotes and say nothing else should be written;
- a hint of the story's question or tension, without spoiling the ending.

Check the title spelling, the faces against the references and that no other text or logo appeared. Generate
a few candidates if needed, keep the rejected ones in `refs/candidates/` and record the prompt and job in
`jobs.json`. The same `cover.png` is the publishing thumbnail/poster.

**Animated opening.** Animate the cover in first-frame mode as `prompts/video/S00_opening.txt`, saved as
`clips/S00_opening.mp4`:

```bash
curl -fsS -X POST "$STUDIO_URL/api/jobs" \
  -F 'prompt=<prompts/video/S00_opening.txt' \
  -F resolution=768p -F lora=<fl2va choice of the film's model family> -F aspect_ratio=image -F duration=5 \
  -F image=@cover.png
```

Use the first-frame (fl2va) choice of the same model family as the film so the look matches, for example
`pika-flex-fl2va-sparse-all` for a Pika H3 Flex film, `fasth3-v2` for a FastH3 film. `duration` is 3–5 seconds;
the frame count snaps to the 17k+5 grid as usual. The prompt starts from `<Picture 1>` (the server adds the
keyframe tag if it is missing), describes the cover in one sentence and then one continuous shot with no cut:

- a very slow, smooth push-in of a few percent;
- gentle ambient motion that belongs to the picture (steam, mist, leaves, light flicker, hair or fabric in the
  wind) and at most one small character gesture such as lifting the eyes;
- the title lettering stays perfectly still, sharp and legible for the whole shot, never morphing or moving
  with the scene, and no other text appears;
- `overall_soundscape`: one opening hit (bell, gong, chord or a sound from the story's world) over quiet
  ambience, no speech; `non_diegetic_music`: a short opening sting in the film's musical idiom.

Review it like any segment, with extra attention to the title: redo once with less motion if the lettering
warps, flickers or gains extra characters. Check its loudness against the first story segment and raise the
opening's audio during assembly if it is noticeably quieter.

## 10. Assemble the entire film

Assemble `clips/S00_opening.mp4` followed by all accepted segments in story order into
`<title>.clean.mp4`, at 24 fps. Normalize frame timestamps
with `setpts=N/24/TB`, set square pixels, and pad/trim each segment's audio to its **video frame duration**
before concat. Re-encode picture/audio for this assembly; concat demuxer plus stream-copy previously retained
AAC priming and caused sync errors. Avoid per-shot loudness normalization, which can lift quiet noise floors.

The helper should use an argument array (for example Python `subprocess.run([...], check=True)`) so paths with
spaces are safe. For segment `i`, compute `d_i = counted_video_frames / 24` and build:

```text
[i:v]setpts=N/24/TB,setsar=1[v_i];
[i:a]asetpts=PTS-STARTPTS,apad=whole_dur=d_i,atrim=0:d_i[a_i];
...
[v_0][a_0][v_1][a_1]...concat=n=SEGMENT_COUNT:v=1:a=1[v][a]
```

These are filter templates: substitute numeric indexes, durations and count before running ffmpeg.
Use `-fps_mode cfr -r 24 -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -c:a aac -b:a 192k
-movflags +faststart`. Do not rebuild the final movie from already encoded preview assemblies.

Verify with ffprobe: 24/1 frame rate, final frame count equals the accepted segments' sum, expected
resolution/orientation, and matching audio/video duration within codec/container rounding. Watch the joins
for rhythm and continuity. Finish the edit **before** transcription so subtitle timestamps follow the deliverable.

## 11. Transcribe the assembled film with Qwen3-ASR

Use the Studio **transcription API with `qwen3_asr`**, not faster-whisper, external TTS or a script-derived
subtitle track. Submit the completed film (or losslessly extracted audio from that exact edit) through the
normal worker queue. Studio manages the ComfyUI/native-audio GPU handoff; no separate manual Whisper
environment or `/api/unload` step is required for this route.

```bash
curl -fsS -X POST "$STUDIO_URL/api/jobs" \
  -F kind=asr -F audio_model=qwen3_asr -F language=Chinese -F timestamps=word \
  -F 'prompt=Final film transcription' -F 'media=@<title>.clean.mp4'
```

Substitute the source language and real filename. The ASR `prompt` is a job label, not an undocumented
vocabulary-hint parameter. Poll the returned job ID. Download `transcript_urls.json`, `.words`, `.srt` and
`.txt` when present (the actual keys are `json`, `words`, `srt`, `txt`) and retain raw results in `subs/`.
The JSON also appears under `result.transcript`. Qwen uses its forced aligner for word timing; exported
cues must be based on real aligned spans. Never invent word timestamps by evenly spreading characters.

The current upload limit is 7200 seconds. If the film exceeds it, extract consecutive audio chunks from the
**final edit**, preserve their exact start offsets, transcribe through the same API and combine timings.
If the whole-film result has local alignment problems, re-transcribe the affected final-edit interval and
apply its exact offset rather than reverting to approximate script timing. If timed exports are missing,
investigate the ASR/alignment failure before delivering subtitles.

## 12. Review, correct and translate subtitles

The agent reviews the transcript itself against the final clip sequence, dialogue/narration script, names,
context and available audio evidence. Fix unreasonable recognition errors, punctuation, homophones,
proper names, repetitions and silence hallucinations. The transcript describes what was actually spoken:
do not restore a scripted line the model omitted, replace an audible paraphrase with the planned wording,
or invent speech to fill a quiet passage. Flag unresolved ambiguity and recheck that interval. Regenerate
a segment if essential story information was never spoken, then reassemble and refresh affected timings.

Save `subs/source.reviewed.srt` and a correction log containing original text, corrected text, time and reason.
Keep trustworthy word/segment boundaries, split at sentence/phrase boundaries and meaningful pauses,
and check subtitle onset/offset at scene changes. No negative, reversed, overlapping accidental or out-of-range
cues; keep readable line lengths and display times. Avoid exposing internal speaker IDs as subtitle text.

**Translate the reviewed transcript yourself** into the user's requested subtitle language, using the story
and character context. Keep names, terminology, tone and implied meaning consistent. Do not translate the
unreviewed ASR or ask ASR to invent a translation. Save `subs/translation.<language>.srt` with matching cue IDs
and times; adapt wording and line breaks for readability without changing meaning. Review both tracks for
omissions, hallucinations, name consistency, punctuation and the chosen target language.

## 13. Embed two selectable subtitle tracks and deliver

Embed the source and translation as **two separate soft subtitle streams**, never burn text into pixels.
Standard players usually display one selected stream at a time; two embedded tracks do not imply simultaneous
bilingual display. If the user requests simultaneous display, clarify that additional presentation requirement.

Example for Chinese source and English translation (replace languages, titles and default selection with
the confirmed settings):

```bash
ffmpeg -i '<title>.clean.mp4' -i subs/source.reviewed.srt -i subs/translation.en.srt \
  -map 0:v:0 -map 0:a:0 -map 1:0 -map 2:0 \
  -c:v copy -c:a copy -c:s mov_text \
  -metadata:s:s:0 language=zho -metadata:s:s:0 title='Chinese (Original)' \
  -metadata:s:s:1 language=eng -metadata:s:s:1 title='English' \
  -disposition:s:0 default -disposition:s:1 0 \
  -movflags +faststart '<title>.mp4'
```

Use a different output path from the input. Verify with ffprobe that there are exactly the intended video,
audio and two subtitle streams, correct language tags/titles/default disposition, unchanged video frame count
and duration, and no subtitle extending past the edit. Check switching and hiding subtitles in a player,
and inspect several cues near the beginning, middle, end and cuts for sync. Keep both SRTs for platforms
that accept sidecar subtitles. Subtitle muxing must not re-encode the final picture/audio.

Use `cover.png` from section 9 as the poster/thumbnail; it is the one image allowed to carry the title, since
story shots have no burned-in text. Write `publish.md` with title, concise description, AI-generation disclosure,
8–12 appropriate hashtags, runtime, resolution and subtitle languages.

Deliver the film path, cover, publish notes and both SRTs, plus a brief account of actual QC and any unresolved
issues. Update `PROGRESS.md` and the job log so another session can reproduce or revise any segment.
