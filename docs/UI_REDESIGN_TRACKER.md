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

| # | Workstream | Status |
|---|-----------|--------|
| 0 | Design tokens + global QSS (`QTstyle.py`) | ✅ Done (verified) |
| 1 | Sidebar labels + active state (`QTabNavigation.py`) | ✅ Done (verified) |
| 2 | Process page: preview-on-top + compare modes | ✅ Done (verified; source-reader compare pending live-render check) |
| 2b | Render queue overhaul | ✅ Done (15 model tests green; live render check pending) |
| 3 | Settings page cards | 🚧 Partial — token styling + accent controls landed; card grouping deferred |
| 4 | Home page hero + CTA | ✅ Done (accent CTA, dev info trimmed, latest-release changelog) |
| 5 | Dead page cleanup | 🚧 Partial — legacy download panel hidden; blank More page + .ui regeneration deferred |
| 6 | Verify loop | 🚧 Screenshots verified; 306 tests green |

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
- [ ] Remove stale `pytorch_version`/`pytorch_backend` combo wiring (flagged tech debt)

## 4. Home page — `apps/gui/pages/home/home.py`

- [ ] Hero row: logo + title + version badge + **Start Enhancing** CTA → Process page
- [ ] System info compact card, user-relevant lines only (drop Python version, CWD)
- [ ] Changelog: latest release only + "show all updates" expander

## 5. Dead page cleanup

- [ ] Remove blank "More" placeholder page from the stack
- [ ] Remove legacy "Backends" panel (stale PyTorch labels) — the download page
      post-ONNX-only already covers updates/uninstall
- [ ] Fix sidebar/nav mapping to match

## 6. Verify loop

- [ ] Screenshot checkpoint after each workstream (offscreen grab harness, reusable)
- [ ] GUI smoke test (`apps/gui/tests/test_gui_smoke.py`) stays green; update if it
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
