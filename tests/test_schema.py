from pathlib import Path

from ui_scene.data.schema import SampleRecord


def test_sample_record_fields() -> None:
    record = SampleRecord(
        sample_id="1",
        label="Video Player",
        split="train",
        image_path=Path("a.jpg"),
        json_path=Path("a.json"),
    )
    assert record.sample_id == "1"
    assert record.label == "Video Player"

