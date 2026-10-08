"""
Tests for the overhauled render queue model.

The product promise: jobs can be added while a render is running, the worker
pulls dynamically (take_next), the running job can't be removed, failed jobs
can be retried, and finishing keeps DONE/FAILED rows visible.
"""

import pytest

from apps.gui.lib.RenderQueue import (
    RenderOptions,
    RenderQueue,
    QUEUED,
    RENDERING,
    DONE,
    FAILED,
    CANCELLED,
)


def make_options(name="clip.mp4"):
    return RenderOptions(
        inputFile=f"/tmp/{name}",
        outputPath=f"/tmp/out_{name}",
        videoWidth=64,
        videoHeight=64,
        videoFps=8,
        tilingEnabled=False,
        tilesize="0",
        videoFrameCount=8,
        backend="onnx",
        modelScale=2,
        interpolateTimes=1,
        benchmarkMode=False,
        sloMoMode=False,
        dyanmicScaleOpticalFlow=False,
        ensemble=False,
    )


@pytest.fixture
def queue():
    q = RenderQueue()
    q.add(make_options("a.mp4"))
    q.add(make_options("b.mp4"))
    return q


class TestDynamicConsumption:
    def test_take_next_marks_rendering(self, queue):
        job = queue.take_next()
        assert job is not None
        assert job.status == RENDERING
        assert queue.jobs[1].status == QUEUED

    def test_take_next_returns_none_when_drained(self, queue):
        queue.take_next()
        queue.take_next()
        assert queue.take_next() is None

    def test_job_added_during_render_is_picked_up(self, queue):
        """The headline behavior: enqueue while a render is running."""
        first = queue.take_next()  # worker starts job A
        queue.add(make_options("c.mp4"))  # user adds mid-render
        assert first is not None
        second = queue.take_next()  # worker finishes A, pulls B
        third = queue.take_next()  # then pulls the newly added C
        assert second.input_file.endswith("b.mp4")
        assert third.input_file.endswith("c.mp4")
        assert third.status == RENDERING

    def test_add_to_start_jumps_the_queue(self, queue):
        queue.addToStart(make_options("urgent.mp4"))
        job = queue.take_next()
        assert job.input_file.endswith("urgent.mp4")


class TestJobStates:
    def test_remove_running_job_is_refused(self, queue):
        queue.take_next()
        queue.remove(0)
        assert len(queue.jobs) == 2

    def test_remove_queued_job(self, queue):
        queue.remove(1)
        assert len(queue.jobs) == 1

    def test_failed_job_can_retry(self, queue):
        job = queue.take_next()
        queue.update_job(job, status=FAILED, return_code=1)
        queue.retry(0)
        assert queue.jobs[0].status == QUEUED
        retried = queue.take_next()
        assert retried is job
        assert retried.status == RENDERING

    def test_cancelled_job_can_retry(self, queue):
        job = queue.take_next()
        queue.update_job(job, status=CANCELLED)
        queue.retry(0)
        assert queue.jobs[0].status == QUEUED

    def test_stop_keeps_remaining_jobs_queued(self, queue):
        """Stop kills the current render but leaves the rest of the queue."""
        first = queue.take_next()
        queue.update_job(first, status=CANCELLED)
        remaining = queue.jobs[1]
        assert remaining.status == QUEUED

    def test_completion_keeps_finished_rows_visible(self, queue):
        """No more queue.clear() at the end — rows stay with status chips."""
        job = queue.take_next()
        queue.update_job(job, status=DONE, return_code=0)
        job2 = queue.take_next()
        queue.update_job(job2, status=DONE, return_code=0)
        assert len(queue.jobs) == 2
        assert queue.counts()["done"] == 2

    def test_counts(self, queue):
        queue.take_next()
        counts = queue.counts()
        assert counts["rendering"] == 1
        assert counts["queued"] == 1


class TestProgressAndSignals:
    def test_job_updated_signal_emits(self, queue):
        seen = []
        queue.jobUpdated.connect(lambda i: seen.append(i))
        job = queue.take_next()
        assert seen == [0]

    def test_queue_changed_signal_emits_on_add(self, queue):
        seen = []
        queue.queueChanged.connect(lambda: seen.append(1))
        queue.add(make_options("x.mp4"))
        assert len(seen) == 1

    def test_set_job_progress(self, queue):
        job = queue.take_next()
        queue.set_job_progress(job, 42, max_value=100)
        assert job.progress == 42
        assert job.max_value == 100

    def test_summary_describes_models(self):
        options = make_options()
        options.upscaleModelFile = "2x_span.onnx"
        options.upscaleModelArch = "SPAN"
        options.interpolateModelFile = "rife.onnx"
        options.interpolateTimes = 2
        from apps.gui.lib.RenderQueue import RenderJob

        job = RenderJob(options)
        assert "SPAN" in job.summary()
        assert "interp 2x" in job.summary()
