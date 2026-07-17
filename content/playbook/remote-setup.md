# Remote Interview Setup

<div class="tag-row"><span class="tag">environment</span><span class="tag">camera / audio</span><span class="tag">CoderPad / live-share</span><span class="tag">backup plans</span><span class="tag">body language on video</span></div>

> [!TIP] Why sweat the setup
> Most RS/AS loops in 2025–2026 are still **virtual** (with some in-person onsites returning). A frozen screen-share, a dead mic, or a dark room converts your content advantage into wasted minutes and interviewer frustration. This is the cheapest edge in the whole process: **you can make the logistics near-perfect with an hour of prep.** Nail it once and forget it for every round.

For a candidate interviewing across time zones (Seoul → US/EU), the setup also has to survive **odd hours and a non-native-English audio channel** — clarity matters more than usual.

## The environment checklist

```mermaid
flowchart LR
  subgraph Space["Physical space"]
    s1[Quiet, door-closable room] --> s2[Neutral, tidy background]
    s2 --> s3[Do-Not-Disturb sign / notify housemates]
  end
  subgraph Tech["Tech stack"]
    t1[Wired ethernet or strong Wi-Fi] --> t2[Laptop on power, not battery]
    t2 --> t3[External mic + camera at eye level]
  end
  Space --> Ready((Ready))
  Tech --> Ready
```

<dl class="kv">
<dt>Room</dt><dd>Quiet, door you can close, phone silenced, notifications off (system-wide Do Not Disturb / Focus). Tell anyone in the space you're unreachable for the window.</dd>
<dt>Background</dt><dd>Neutral and tidy beats a virtual background (which glitches around your hands — bad when you gesture at a whiteboard). A plain wall or bookshelf is ideal.</dd>
<dt>Power & network</dt><dd>Laptop plugged in. **Wired ethernet** if possible; otherwise sit close to the router. Close bandwidth hogs (cloud sync, downloads, other tabs).</dd>
</dl>

## Camera & audio: the details that read as "professional"

Audio quality matters more than video — a crisp mic keeps a non-native accent perfectly intelligible; a laptop mic in a echoey room does not.

| Element | Cheap fix | Why it matters |
| --- | --- | --- |
| **Mic** | Wired earbuds/headset mic > laptop mic | kills room echo & keyboard clatter; clearer for non-native accents |
| **Camera height** | Raise laptop so the lens is at **eye level** | looking down reads as low-confidence; eye-level = engaged |
| **Lighting** | Face a window or put a lamp *behind the camera* | back-lighting turns you into a silhouette |
| **Framing** | Head-and-shoulders, small headroom | too close = intense; too far = disengaged |
| **Eye contact** | Glance at the **lens**, not the face on screen | on video, lens-gaze *is* eye contact |

> [!WARNING] Test with the actual tool, not just a mirror
> "Camera works" in Photo Booth ≠ "works in the interview app." Do a real test call (a friend, or the platform's test room) on the **same app, same machine, same network** you'll use. Verify the mic isn't the wrong device and screen-share actually transmits.

## Practice the coding/collaboration tool in advance

The interview platform is a variable you can *remove*. Each has quirks — no autocomplete, no run button, unfamiliar keybindings — that eat time if you meet them cold.

<dl class="kv">
<dt>CoderPad / Codility / HackerRank</dt><dd>Shared editor, often minimal. Practice: writing without IDE autocomplete, using its language selector, running tests if allowed. Meta's phone screen is typically CoderPad.</dd>
<dt>VS Code Live Share</dt><dd>Real-time collaborative editing in your own editor. Install and test it beforehand; know how to share terminal and follow the interviewer's cursor.</dd>
<dt>Google Doc / plain shared text</dt><dd>Some research rounds use a bare doc — no execution. Practice writing *runnable-looking* code and tracing by hand, since you can't lean on a compiler.</dd>
<dt>Virtual whiteboard (Excalidraw / built-in)</dt><dd>For system/ML design. Practice drawing boxes-and-arrows fast; know how to make/label/move nodes. See [The Design Framework](#/system-design/framework).</dd>
</dl>

> [!TIP] Ask the recruiter which tool
> Include it in your recruiter-screen question list (see [Recruiter & HM Screens](#/process/recruiter-hm)): *"Which coding platform will we use, and is code execution / AI assistance allowed?"* Then do one warm-up problem in that exact tool.

## AI-assisted coding rounds (2026)

Some companies (notably Meta) now run **AI-assisted coding rounds** where an assistant/autocomplete is allowed or expected. Confirm the policy with the recruiter and, if allowed, practice the *workflow*: prompt for scaffolding, then read, verify, and edit critically. The signal being tested shifts from "can you recall syntax" to "can you steer and vet a tool" — narrate your verification, don't blindly accept output.

## Backup plans (things *will* break)

Have a written fallback so a glitch costs 30 seconds, not the round.

```mermaid
flowchart TB
  P["Problem hits"] --> A{"What broke?"}
  A -->|Internet drops| B["Switch to phone hotspot<br/>(pre-configured)"]
  A -->|App/video freezes| C["Rejoin from a second device<br/>(link open on phone)"]
  A -->|Audio fails| D["Dial in by phone<br/>(number saved) / type in chat"]
  A -->|Power/laptop dies| E["Second charged device ready"]
  B --> R["Message recruiter/interviewer<br/>on the saved channel"]
  C --> R
  D --> R
  E --> R
```

- **Save the interviewer/recruiter contact** (email + any phone/Slack) *before* the call, so you can reach them instantly if you drop.
- **Pre-configure a phone hotspot** as a network fallback.
- **Keep the join link open on a second device** so you can rejoin in seconds.
- If something breaks, **stay calm and communicate**: "I've lost audio — dialing in now." Composure under a glitch is itself a positive signal.

## Body language on video

Video flattens presence, so compensate slightly.

<div class="proscons"><div><div class="pros-t">Reads well</div>

- Lens-level gaze; look at the camera when *you* speak
- Sit up, shoulders back, slight lean-in when listening
- Hands visible; natural gestures for emphasis
- Nod/verbal "mm-hm" to show you're tracking
- Smile at greeting and sign-off

</div><div><div class="cons-t">Reads poorly</div>

- Staring at your own thumbnail (looks like avoidance)
- Slouching / leaning out of frame
- Reading obviously off a second screen
- Flat, motionless "hostage video" energy
- Fidgeting, off-screen glances (looks like notetaking-cheating)

</div></div>

> [!NOTE] Notes on video are fine — if visible-ish
> Glancing at a small note card is normal, but *obvious* long reads off-screen look like you're reading answers. Keep any notes to keyword bullets (your [story-bank](#/behavioral/star) triggers, questions to ask), not scripts. More in [Day-Of Tactics](#/playbook/tactics).

## The 30-minute-before ritual

- [ ] Restart the machine; close every non-essential app and tab.
- [ ] System Do Not Disturb / Focus on; phone silent & face-down.
- [ ] Test camera, mic, speaker in the **actual** app; join link ready on a 2nd device.
- [ ] Water within reach; bathroom done; snack if it's a long loop.
- [ ] Interviewer names & the recruiter contact open in a tab.
- [ ] Résumé, JD, and your story-bank keywords visible but minimal.
- [ ] Join **2–3 minutes early**; sit ready, don't scramble at 0:00.

## Follow-ups

<details class="qa"><summary>My internet is unreliable — should I flag it?</summary>
<div class="qa-body">

**Short:** Yes, proactively, and have the hotspot ready.

**Deep:** A one-line heads-up at the start ("my connection has been slightly flaky today; if I freeze I'll rejoin immediately from my phone") sets expectations and reads as prepared, not fragile. Then if it happens, you execute the plan calmly instead of apologizing in a panic.
</div></details>

<details class="qa"><summary>Can I use a virtual background or blur?</summary>
<div class="qa-body">

**Short:** Prefer a real tidy background; light blur is acceptable, full virtual backgrounds are risky.

**Deep:** Virtual backgrounds glitch around moving hands and gestures — distracting in a design round where you point at a shared board. If your space is messy, a *subtle* blur is the safe middle. Never a distracting themed background.
</div></details>

## Cheat-sheet

| Area | Must-do |
| --- | --- |
| Network | wired/strong Wi-Fi; laptop on power; hotspot fallback ready |
| Audio | headset mic > laptop mic; test in the real app |
| Camera | lens at eye level; light in front, not behind; head-and-shoulders |
| Tool | ask recruiter which platform; do 1 warm-up in it; know its quirks |
| AI rounds | confirm policy; if allowed, practice steer-and-verify workflow |
| Backup | save contacts; 2nd device on join link; calm comms if it breaks |
| Body language | look at the lens; sit up; hands visible; not at your own thumbnail |
| Ritual | restart, DND, test, water, names & notes up, join 2–3 min early |

**Related:** [Communication & Whiteboarding](#/playbook/communication) · [Day-Of Tactics & Recovery](#/playbook/tactics) · [Recruiter & HM Screens](#/process/recruiter-hm) · [Coding Round Strategy](#/coding/strategy) · [The Design Framework](#/system-design/framework) · [Questions to Ask Them](#/playbook/questions-to-ask)
