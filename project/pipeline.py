"""Top-level dataset engineering runner."""

from dataset_engineering.pipeline import DatasetEngineeringPipeline


def main() -> None:
    """Execute the dataset engineering workflow."""
    DatasetEngineeringPipeline().run()


if __name__ == "__main__":
    main()
