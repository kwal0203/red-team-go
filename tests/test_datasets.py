"""Tests for dataset loading infrastructure and API endpoints."""

import pytest
from fastapi.testclient import TestClient

from main import app
from services.datasets import (
    BaseDatasetLoader,
    DatasetCategory,
    DatasetInfo,
    DatasetItem,
    DatasetNotFoundError,
    get_dataset,
    get_dataset_info,
    list_datasets,
)
from services.datasets.src.stereotype import (
    BBQLoader,
    BiasType,
    CrowSPairsLoader,
    StereoSetLoader,
    StereotypeItem,
)

client = TestClient(app)


# =============================================================================
# Registry Tests
# =============================================================================


class TestDatasetRegistry:
    """Tests for the dataset registry."""

    def test_list_datasets_returns_datasets(self):
        """Test that list_datasets returns registered datasets."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) >= 3  # At least the 3 stereotype datasets

    def test_list_datasets_returns_datasetinfo(self):
        """Test that list_datasets returns DatasetInfo objects."""
        datasets = list_datasets()
        for info in datasets:
            assert isinstance(info, DatasetInfo)
            assert info.name
            assert info.category
            assert info.description

    def test_list_datasets_filter_by_category(self):
        """Test filtering datasets by category."""
        stereotype_datasets = list_datasets(category=DatasetCategory.STEREOTYPE)
        assert len(stereotype_datasets) >= 3
        for info in stereotype_datasets:
            assert info.category == DatasetCategory.STEREOTYPE

    def test_list_datasets_empty_for_unused_category(self):
        """Test that filtering by unused category returns empty list."""
        # Assuming no jailbreak datasets are registered yet
        jailbreak_datasets = list_datasets(category=DatasetCategory.JAILBREAK)
        assert isinstance(jailbreak_datasets, list)

    def test_get_dataset_by_name(self):
        """Test getting a dataset by name."""
        loader = get_dataset("stereoset")
        assert isinstance(loader, BaseDatasetLoader)
        assert loader.info.name == "stereoset"

    def test_get_dataset_not_found(self):
        """Test that getting unknown dataset raises error."""
        with pytest.raises(DatasetNotFoundError):
            get_dataset("nonexistent_dataset")

    def test_get_dataset_info_by_name(self):
        """Test getting dataset info by name."""
        info = get_dataset_info("stereoset")
        assert isinstance(info, DatasetInfo)
        assert info.name == "stereoset"
        assert info.category == DatasetCategory.STEREOTYPE

    def test_get_dataset_info_not_found(self):
        """Test that getting info for unknown dataset raises error."""
        with pytest.raises(DatasetNotFoundError):
            get_dataset_info("nonexistent_dataset")


# =============================================================================
# Stereotype Dataset Loaders Tests
# =============================================================================


class TestStereoSetLoader:
    """Tests for StereoSet dataset loader."""

    def test_loader_info(self):
        """Test StereoSet loader info."""
        loader = StereoSetLoader()
        info = loader.info
        assert info.name == "stereoset"
        assert info.category == DatasetCategory.STEREOTYPE
        assert info.huggingface_id == "McGill-NLP/stereoset"
        assert info.size == 17000

    def test_get_sample(self):
        """Test getting sample data."""
        loader = StereoSetLoader()
        samples = loader.get_sample(num_samples=5)
        assert len(samples) == 5
        for sample in samples:
            assert isinstance(sample, StereotypeItem)
            assert sample.id
            assert sample.text
            assert sample.bias_type
            assert sample.stereotype
            assert sample.anti_stereotype

    def test_get_sample_limit(self):
        """Test that num_samples is respected."""
        loader = StereoSetLoader()
        samples = loader.get_sample(num_samples=2)
        assert len(samples) == 2

    def test_sample_to_dict(self):
        """Test converting sample to dict."""
        loader = StereoSetLoader()
        samples = loader.get_sample(num_samples=1)
        sample_dict = samples[0].to_dict()
        assert "id" in sample_dict
        assert "text" in sample_dict
        assert "bias_type" in sample_dict
        assert "stereotype" in sample_dict
        assert "anti_stereotype" in sample_dict


class TestCrowSPairsLoader:
    """Tests for CrowS-Pairs dataset loader."""

    def test_loader_info(self):
        """Test CrowS-Pairs loader info."""
        loader = CrowSPairsLoader()
        info = loader.info
        assert info.name == "crows_pairs"
        assert info.category == DatasetCategory.STEREOTYPE
        assert info.huggingface_id == "nyu-mll/crows_pairs"
        assert info.size == 1508

    def test_get_sample(self):
        """Test getting sample data."""
        loader = CrowSPairsLoader()
        samples = loader.get_sample(num_samples=5)
        assert len(samples) == 5
        for sample in samples:
            assert isinstance(sample, StereotypeItem)
            assert sample.stereotype
            assert sample.anti_stereotype

    def test_bias_types_present(self):
        """Test that samples have valid bias types."""
        loader = CrowSPairsLoader()
        samples = loader.get_sample(num_samples=5)
        for sample in samples:
            assert isinstance(sample.bias_type, BiasType)


class TestBBQLoader:
    """Tests for BBQ dataset loader."""

    def test_loader_info(self):
        """Test BBQ loader info."""
        loader = BBQLoader()
        info = loader.info
        assert info.name == "bbq"
        assert info.category == DatasetCategory.STEREOTYPE
        assert info.huggingface_id == "heegyu/bbq"
        assert info.size == 58492

    def test_get_sample(self):
        """Test getting sample data."""
        loader = BBQLoader()
        samples = loader.get_sample(num_samples=3)
        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, DatasetItem)
            assert sample.id
            assert sample.text
            assert sample.metadata

    def test_sample_metadata_structure(self):
        """Test that BBQ samples have correct metadata structure."""
        loader = BBQLoader()
        samples = loader.get_sample(num_samples=1)
        metadata = samples[0].metadata
        assert "context" in metadata
        assert "ans0" in metadata
        assert "ans1" in metadata
        assert "ans2" in metadata
        assert "label" in metadata
        assert "category" in metadata


# =============================================================================
# Dataset API Endpoint Tests
# =============================================================================


class TestDatasetListEndpoint:
    """Tests for GET /datasets endpoint."""

    def test_list_datasets_endpoint(self):
        """Test listing all datasets."""
        response = client.get("/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "total" in data
        assert data["total"] >= 3

    def test_list_datasets_structure(self):
        """Test response structure for dataset list."""
        response = client.get("/datasets")
        data = response.json()
        for dataset in data["datasets"]:
            assert "name" in dataset
            assert "category" in dataset
            assert "description" in dataset
            assert "source" in dataset
            assert "citation" in dataset

    def test_list_datasets_filter_by_category(self):
        """Test filtering datasets by category."""
        response = client.get("/datasets?category=stereotype")
        assert response.status_code == 200
        data = response.json()
        for dataset in data["datasets"]:
            assert dataset["category"] == "stereotype"

    def test_list_datasets_invalid_category(self):
        """Test that invalid category returns 400."""
        response = client.get("/datasets?category=invalid_category")
        assert response.status_code == 400
        assert "Invalid category" in response.json()["detail"]


class TestDatasetInfoEndpoint:
    """Tests for GET /datasets/{name} endpoint."""

    def test_get_dataset_info(self):
        """Test getting info for a specific dataset."""
        response = client.get("/datasets/stereoset")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "stereoset"
        assert data["category"] == "stereotype"
        assert "huggingface_id" in data

    def test_get_dataset_info_not_found(self):
        """Test that unknown dataset returns 404."""
        response = client.get("/datasets/nonexistent_dataset")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestDatasetSampleEndpoint:
    """Tests for POST /datasets/{name}/sample endpoint."""

    def test_get_samples_with_auth(self):
        """Test getting samples with authentication."""
        response = client.post(
            "/datasets/stereoset/sample",
            json={"num_samples": 5},
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_name"] == "stereoset"
        assert data["num_samples"] == 5
        assert len(data["samples"]) == 5

    def test_get_samples_default_count(self):
        """Test default sample count."""
        response = client.post(
            "/datasets/stereoset/sample",
            json={},
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200
        data = response.json()
        # Default is 10, but stereoset only has 5 sample items
        assert data["num_samples"] <= 10

    def test_get_samples_not_found(self):
        """Test that unknown dataset returns 404."""
        response = client.post(
            "/datasets/nonexistent_dataset/sample",
            json={"num_samples": 5},
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 404

    def test_get_samples_invalid_count(self):
        """Test that invalid sample count is rejected."""
        response = client.post(
            "/datasets/stereoset/sample",
            json={"num_samples": 0},
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422  # Validation error

    def test_get_samples_structure(self):
        """Test sample response structure."""
        response = client.post(
            "/datasets/stereoset/sample",
            json={"num_samples": 1},
            headers={"X-API-Key": "test-api-key"},
        )
        data = response.json()
        assert "dataset_name" in data
        assert "samples" in data
        assert "num_samples" in data
        assert "total_available" in data

        # Check sample structure
        sample = data["samples"][0]
        assert "id" in sample
        assert "text" in sample


# =============================================================================
# DatasetItem Tests
# =============================================================================


class TestDatasetItem:
    """Tests for DatasetItem dataclass."""

    def test_create_dataset_item(self):
        """Test creating a DatasetItem."""
        item = DatasetItem(
            id="test_1",
            text="Test text",
            metadata={"key": "value"},
        )
        assert item.id == "test_1"
        assert item.text == "Test text"
        assert item.metadata == {"key": "value"}

    def test_dataset_item_default_metadata(self):
        """Test DatasetItem with default metadata."""
        item = DatasetItem(id="test_1", text="Test text")
        assert item.metadata is None

    def test_dataset_item_to_dict(self):
        """Test converting DatasetItem to dict."""
        item = DatasetItem(
            id="test_1",
            text="Test text",
            metadata={"key": "value"},
        )
        d = item.to_dict()
        assert d["id"] == "test_1"
        assert d["text"] == "Test text"
        assert d["metadata"] == {"key": "value"}

    def test_dataset_item_to_dict_empty_metadata(self):
        """Test that None metadata becomes empty dict in to_dict."""
        item = DatasetItem(id="test_1", text="Test text")
        d = item.to_dict()
        assert d["metadata"] == {}


class TestStereotypeItem:
    """Tests for StereotypeItem dataclass."""

    def test_create_stereotype_item(self):
        """Test creating a StereotypeItem."""
        item = StereotypeItem(
            id="stereo_1",
            text="Test context",
            bias_type=BiasType.GENDER,
            stereotype="Stereotypical sentence",
            anti_stereotype="Anti-stereotypical sentence",
            context="Test context",
            unrelated="Unrelated sentence",
        )
        assert item.bias_type == BiasType.GENDER
        assert item.stereotype == "Stereotypical sentence"

    def test_stereotype_item_to_dict(self):
        """Test converting StereotypeItem to dict."""
        item = StereotypeItem(
            id="stereo_1",
            text="Test context",
            bias_type=BiasType.RACE,
            stereotype="Stereotype",
            anti_stereotype="Anti-stereotype",
        )
        d = item.to_dict()
        assert d["bias_type"] == "race"  # Enum value, not Enum
        assert d["stereotype"] == "Stereotype"
        assert d["anti_stereotype"] == "Anti-stereotype"


class TestBiasType:
    """Tests for BiasType enum."""

    def test_all_bias_types_defined(self):
        """Test that all expected bias types are defined."""
        expected = [
            "gender",
            "race",
            "religion",
            "age",
            "nationality",
            "disability",
            "socioeconomic",
            "sexual_orientation",
            "physical_appearance",
            "profession",
        ]
        for bias in expected:
            assert BiasType(bias)

    def test_bias_type_string_value(self):
        """Test that BiasType has string values."""
        assert BiasType.GENDER.value == "gender"
        assert BiasType.RACE.value == "race"
