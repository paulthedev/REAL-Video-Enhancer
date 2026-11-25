# UI Redesign Tracker — "Studio Polish"

Design direction: **Studio polish** (keep page structure/flow, rebuild layouts) with a
**cyan/teal accent (#22D3EE)** on the dark theme. Decided 2026-10-08.

Scope anchor (from the approved outline):

- Process page: live preview display **on top** with **Enhanced / Side-by-Side /
  Before⟷After slider** view modes, options **below it**
- Queue overhaul: adding videos to the queue stays possible **while a render is running**
- No light theme, no new features beyond the compare modes + queue work, preview
  playback behavior unchanged, and the flat `self.<widget>` name contract preserved

## Status Overview

> Historical checklist. The workstreams shipped in the October–November
> 2026 commits; **unchecked bullets below are not open work** — they are
> retained only as the original design notes. Remaining deliberate follow-ups
> are called out in the table.

| # | Workstream | Status |
|---|-----------|--------|
| 0 | Design tokens + global QSS (`QTstyle.py`) | ✅ Done (verified) |
| 1 | Sidebar labels + active state (`QTabNavigation.py`) | ✅ Done (verified) |
| 2 | Process page: preview-on-top + compare modes | ✅ Done (verified) |
| 2b | Render queue overhaul | ✅ Done (15 model tests green) |
| 3 | Settings page cards | 🚧 Partial — token styling + accent controls landed; card grouping deferred |
| 4 | Home page hero + CTA | ✅ Done (accent CTA, dev info trimmed, latest-release changelog) |
| 5 | Dead page cleanup | ✅ Done — blank More page + legacy Backends panel removed; Models page is the custom-model import hub |
| 6 | Verify loop | ✅ Screenshots verified; GUI smoke + runtime tests green |

## Post-audit fixes (2026-11-26)

An offscreen visual/runtime audit at 1440×900 and 960×620 closed the
functional UI issues:

- Preview no longer crashes with no video and no longer clears the render
  queue before validation.
- Home changelog loads off the GUI thread with an offline fallback; the orphan
  “Support Me!” caption is gone and the hero title sits beside the logo.
- Dead controls hidden: the `onnx`-only Backend combo, Interpolation UHD Mode,
  and the leftover torch.compile help icon.
- Pause/Stop are hidden while idle and appear only during a render.
- Models page: imported-model list, status under the drop zone, legacy empty
  download area hidden, uninstall moved to a danger zone.
- Source inputs and the Logs tab have empty-state text; the duplicate queue
  empty-state line is gone; scene-threshold moved to Render Settings.
- Encoder command field is labeled, checkbox rows are fully clickable, and
  the process tabs fit at the 960 px minimum width.
- Source reader is stopped on close (no more QThread shutdown warning).
- Notifications are compact status chips instead of a full-window dim.
- The GUI render flow works from a source checkout (backend path), the live
  preview paints all three modes without QPainter errors, queue rows reflow
  with the rail width, and footer counts refresh as jobs finish.
- Closing the window only asks for confirmation when a render/queued job is
  active.

Remaining polish (deliberately not changed): the Advanced/RVE settings cards
are sparse, and the Models page keeps a large empty area below the list.

---

## 0. Design tokens + global QSS — `apps/gui/lib/QTstyle.py`

- [ ] Token set: accent `#22D3EE` (+ hover/dim variants), background layers
      (window `#0f1216` → sidebar `#16191d` → card `#1c2128` → field `#262c35`),
      text tiers, radius/spacing scales, danger red
- [ ] Primary button (accent-filled, dark text), ghost/secondary buttons
- [ ] Inputs & combos with accent focus ring; pill-style QTabBar; accent checkboxes
- [ ] Slim dark scrollbars; accent progress chunk; styled tooltips
- [ ] All selectors objectName/type-based (no widget renames needed)

## 1. Sidebar — `apps/gui/widgets/QTabNavigation.py`

- [ ] Text labels under icons (Home / Process / Settings / Downloads); widen 72 → ~88px
- [ ] Active state: accent-tinted pill + left accent bar; hover state; tooltips

## 2. Process page — `apps/gui/pages/process/process.py` + new `apps/gui/widgets/PreviewCompare.py`

Layout (preview on top, options below):

```
┌──────────────────────────────────────────────────────┐
│  LIVE PREVIEW (stretches)                            │
│  [ Enhanced | Side by Side | Before ⟷ After ]        │
├──────────────────────────────────────────────────────┤
│  Source card │ Models card │ Queue & Status card     │
├──────────────────────────────────────────────────────┤
│  [ ▶ ENHANCE ]  ▓▓▓▓░░░░ 62% · ETA 04:12             │
└──────────────────────────────────────────────────────┘
```

- [ ] `PreviewCompare.py`: three view modes — Enhanced (streamed frame),
      Side by Side (original | enhanced, captions), Before⟷After (draggable split
      handle); empty-state placeholder ("start a render to see live preview")
- [ ] Source-frame reader (GUI-side cv2 capture on the job input, seek to
      `output frame ÷ interpolate factor`); comparison modes disable with tooltip
      when input is a URL/unreadable
- [ ] `UpdateGUIThread` (QTcustom.py:233) feeds the new widget (replaces the
      currently hidden `VideoPreview` label)
- [ ] Options zone below preview: Source card (input/batch/destination),
      Models card (upscale/interp/denoise/decompress + pickers inline),
      Queue & Status card (see 2b)
- [ ] Full-width accent Enhance CTA + inline progress bar (bottom action bar)

## 2b. Render queue overhaul — `apps/gui/lib/RenderQueue.py`

Pain point being fixed: `disableProcessPage()` (REAL-Video-Enhancer.py:707) disables
the entire process page for the whole batch, and the worker snapshot-iterates the
queue — so nothing can be added once rendering starts.

- [ ] `RenderJob` wrapper around `RenderOptions` with status
      `Queued → Rendering → Done / Failed / Cancelled`
- [ ] Worker loop consumes queue dynamically (finish job → next queued);
      enqueue-during-render works
- [ ] Rendering locks only render controls; input/models/destination stay live
- [ ] Queue panel rows: filename + model summary + status badge + inline progress
      (active row); actions: reorder, remove (queued only), retry (failed),
      open output folder (done); footer: clear finished + counts
- [ ] Overwrite prompts per-job at job start (no batch ask upfront)
- [ ] Contextual render controls: Enhance → ⏸ Pause / ⏹ Stop → Enhance;
      Stop kills current render, leaves remaining jobs queued
- [ ] Old "Render Queue" tab folded into the Queue & Status card
- [ ] Out of scope (deliberate): parallel renders, queue persistence across restarts

## 3. Settings page — `apps/gui/pages/settings/settings.py`

- [ ] Rows grouped into section cards (Output / Scene Detection / Rendering / App)
      with headers; info icons beside labels; Reset Settings in card footer
- [x] Remove stale `pytorch_version`/`pytorch_backend` combo wiring (done 2026-11-25)

## 4. Home page — `apps/gui/pages/home/home.py`

- [ ] Hero row: logo + title + version badge + **Start Enhancing** CTA → Process page
- [ ] System info compact card, user-relevant lines only (drop Python version, CWD)
- [ ] Changelog: latest release only + "show all updates" expander

## 5. Dead page cleanup

- [x] Remove blank "More" placeholder page from the stack
- [x] Remove legacy "Backends" panel (stale PyTorch labels) — the download page
      post-ONNX-only already covers updates/uninstall
- [ ] Fix sidebar/nav mapping to match

## 6. Verify loop

- [ ] Screenshot checkpoint after each workstream (offscreen grab harness, reusable)
- [x] GUI smoke test (`apps/gui/tests/test_gui_smoke.py`) stays green; update if it
      touches removed pages/widgets
- [ ] New regression coverage: add-to-queue-while-rendering behavior

---

## Decisions log

| Date | Decision |
|------|----------|
| 2026-10-08 | Qt cascade gotcha found: generated per-page .ui stylesheets override the window sheet — cleared after assemble_pages (closer ancestor wins regardless of specificity) |
| 2026-10-08 | Fusion + token palette applied in MainWindow.__init__ (not just main()) so offscreen tests/native dialogs match |
| 2026-10-08 | Sidebar top-aligned with stretch; process page rebuilt in code (layout.py) re-homing generated widgets — flat self.<name> contract preserved |
| 2026-10-08 | Queue rows are status-driven (Queued/Rendering/Done/Failed/Cancelled); rendering no longer disables the page; per-job overwrite prompts marshal to the GUI thread via signal+Event |
| 2026-10-08 | Direction: Studio polish (structure kept, layouts rebuilt) over full rebuild or reskin-only |
| 2026-10-08 | Accent: cyan/teal `#22D3EE` |
| 2026-10-08 | Preview goes on top with compare modes (Enhanced / Side-by-Side / Before⟷After); options below |
| 2026-10-08 | Queue must accept new jobs during an active render; per-job overwrite prompts; sequential rendering kept |
| 2026-10-08 | Compare modes need a GUI-side source-frame reader (backend streams only the processed frame) |
