"""Проверки состояния фоновой индексации: что показывает интерфейс."""
import unittest

from core.jobs import Jobs


class JobTests(unittest.TestCase):
    def setUp(self):
        self.jobs = Jobs()

    def test_progress_events_fill_the_job_and_finish_it(self):
        job = self.jobs.start("cooking")
        self.jobs.apply(job, {"event": "files", "total": 3})
        self.jobs.apply(job, {"event": "file", "source": "Плов.pdf",
                              "index": 2, "total": 3})
        self.jobs.apply(job, {"event": "chunks", "saved": 96, "source": "Плов.pdf"})
        self.jobs.apply(job, {"event": "file_done", "source": "Плов.pdf",
                              "chunks": 100})
        self.jobs.apply(job, {"event": "failed", "source": "скан.pdf",
                              "message": "таймаут"})
        self.jobs.apply(job, {"event": "done", "chunks": 500, "files": 3,
                              "failed": 1, "added": 2})

        data = job.as_dict()
        self.assertEqual(data["state"], "done")
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["file"], "Плов.pdf")
        self.assertEqual(data["chunks"], 500)
        self.assertEqual(data["failed"], 1)
        self.assertIsNotNone(data["finished"])
        self.assertGreaterEqual(data["seconds"], 0)
        self.assertTrue(any("Плов.pdf" in line for line in data["log"]))
        self.assertTrue(any("скан.pdf" in line for line in data["log"]))

    def test_chunks_counter_starts_from_zero_for_each_book(self):
        """«в этой книге» — про текущую книгу: у следующей счётчик с нуля."""
        job = self.jobs.start("cooking")
        self.jobs.apply(job, {"event": "file", "source": "первая.pdf",
                              "index": 1, "total": 2})
        self.jobs.apply(job, {"event": "chunks", "saved": 120})
        self.jobs.apply(job, {"event": "file_done", "source": "первая.pdf",
                              "chunks": 120})
        self.jobs.apply(job, {"event": "file", "source": "вторая.pdf",
                              "index": 2, "total": 2})

        self.assertEqual(job.chunks, 0)
        self.assertEqual(job.file, "вторая.pdf")
        self.assertEqual(job.index, 2)

    def test_error_event_marks_the_job_failed(self):
        job = self.jobs.start("cooking")
        self.jobs.apply(job, {"event": "error", "message": "в папке нет книг"})
        self.assertEqual(job.state, "error")
        self.assertEqual(job.error, "в папке нет книг")

    def test_unexpected_failure_is_reported(self):
        job = self.jobs.start("cooking")
        self.jobs.fail(job, "база занята")
        self.assertEqual(job.state, "error")
        self.assertEqual(job.error, "база занята")

    def test_latest_returns_the_newest_job_of_that_agent(self):
        first = self.jobs.start("cooking")
        self.jobs.fail(first, "упало")
        second = self.jobs.start("cooking")
        self.jobs.start("other")

        self.assertIs(self.jobs.latest("cooking"), second)
        self.assertIs(self.jobs.get(second.id), second)
        self.assertIsNone(self.jobs.latest("missing"))

    def test_old_finished_jobs_do_not_pile_up(self):
        jobs = Jobs(keep=2)
        for _ in range(5):
            jobs.fail(jobs.start("cooking"), "нет")
        self.assertEqual(len(jobs._jobs), 2)

    def test_log_keeps_only_the_tail(self):
        job = self.jobs.start("cooking")
        for number in range(30):
            self.jobs.apply(job, {"event": "file", "source": f"книга{number}.pdf",
                                  "index": number, "total": 30})
        self.assertEqual(len(job.as_dict()["log"]), 12)
        self.assertIn("книга29.pdf", job.as_dict()["log"][-1])


if __name__ == '__main__':
    unittest.main()
