import torch
from pymilvus import connections, utility, CollectionSchema, Collection
from typing import List, Dict, Any, Tuple, Union
import logging


class MilvusConnector:
    """
    A connector class for interacting with a Milvus server to manage collections,
    insert data, and perform searches.

    Attributes:
        _host (str): The Milvus server host.
        _port (str): The Milvus server port.
        _vector_dimension (int): The dimension of the vector embeddings.
        _connected (bool): Whether the connection to the server is established.
    """

    def __init__(self, host: str = "localhost", port: str = "19530", vector_dimension: int = 512):
        """
        Initializes the MilvusConnector with server details and vector dimension.

        Args:
            host (str): Milvus server host. Defaults to "localhost".
            port (str): Milvus server port. Defaults to "19530".
            vector_dimension (int): Dimension of the vector embeddings. Defaults to 512.
        """
        self._host = host
        self._port = port
        self._vector_dimension = vector_dimension
        self._connected = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def connect(self) -> None:
        """
        Establishes a connection to the Milvus server.
        """
        if not self._connected:
            try:
                connections.connect(alias="default", host=self._host, port=self._port)
                self._connected = True
            except Exception as e:
                self._logger.error(f"Error connecting to Milvus: {e}")

    def disconnect(self) -> None:
        """
        Disconnects from the Milvus server.
        """
        if self._connected:
            connections.disconnect(alias="default")
            self._connected = False

    def __enter__(self) -> "MilvusConnector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()

    def check_if_collection_exists(self, collection_name: str) -> bool:
        """
        Checks if a collection exists in Milvus.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        return utility.has_collection(collection_name)

    def create_collection(self, collection_name: str, fields: List[Any], remove_if_exists: bool = True) -> None:
        """
        Creates a collection in Milvus with the specified schema.

        Args:
            collection_name (str): Name of the collection to create.
            fields (List[Any]): Schema fields for the collection.
            remove_if_exists (bool): Whether to remove the collection if it already exists. Defaults to True.
        """
        try:
            if remove_if_exists and utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                self._logger.info(f"Existing collection '{collection_name}' dropped.")

            schema = CollectionSchema(fields=fields)
            collection = Collection(name=collection_name, schema=schema)

            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1536},
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
            self._logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            self._logger.error(f"Error creating collection '{collection_name}': {e}")


    def insert(self, data: List[Dict[str, Any]], collection_name: str = "default", batch_size: int = 1000) -> None:
        """
        Inserts data into the specified collection.

        Args:
            data (List[Dict[str, Any]]): List of data dictionaries to insert.
            collection_name (str): Name of the collection to insert data into.
            batch_size (int): Batch size for data insertion. Defaults to 1000.
        """
        try:
            collection = Collection(name=collection_name)
            if not isinstance(data, list):
                raise ValueError("Data must be a list of dictionaries.")

            schema_fields = collection.schema.fields
            required_fields = [
                field.name for field in schema_fields if not field.is_primary or not field.auto_id
            ]

            batch_count = (len(data) + batch_size - 1) // batch_size
            for batch_idx in range(batch_count):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(data))

                batch = data[start_idx:end_idx]
                data_batch = []

                for field in required_fields:
                    if field in batch[0]:
                        data_batch.append([item.get(field, None) for item in batch])
                    else:
                        raise ValueError(f"Field '{field}' is missing in data.")

                insert_result = collection.insert(data_batch)
                self._logger.info(
                    f"Inserted {len(insert_result.primary_keys)} entries into {collection_name} (Batch {batch_idx + 1}/{batch_count})"
                )
            collection.flush()
        except Exception as e:
            self._logger.error(f"Error inserting data: {e}")


    def delete_entity_by_id(self, ids: List[int], collection_name: str = "images") -> bool:
        """
        Deletes entities from a collection by ID.

        Args:
            ids (List[int]): List of IDs to delete.
            collection_name (str): Name of the collection. Defaults to "images".

        Returns:
            bool: True if deletion is successful.
        """
        try:
            collection = Collection(name=collection_name)
            collection.delete(expr=f"id in {str(ids)}")
            collection.flush()
            self._logger.info(f"Entities with IDs {ids} deleted from collection '{collection_name}'.")
            return True
        except Exception as e:
            self._logger.error(f"Error deleting entities with IDs {ids} from collection '{collection_name}': {e}")
            return False

    def search_topk(
        self, search_embedded: Union[torch.Tensor, List[float]], output_field: str, k: int = 3, collection_name: str = "images"
    ) -> List[Tuple[Any, float]]:
        """
        Searches for the top-k closest vectors in a collection.

        Args:
            search_embedded (Union[torch.Tensor, List[float]]): The query vector.
            output_field (str): The field to return in results.
            k (int): Number of top results to retrieve. Defaults to 3.
            collection_name (str): Name of the collection. Defaults to "images".

        Returns:
            List[Tuple[Any, float]]: A list of tuples containing the output_field value and distance.
        """
        collection = Collection(name=collection_name)
        collection.load()

        if isinstance(search_embedded, torch.Tensor):
            search_embedded = search_embedded.tolist()

        return [
            (getattr(r.entity, output_field), r.distance)
            for r in collection.search(
                data=search_embedded,
                anns_field="embedding",
                param={},
                limit=k,
                output_fields=[output_field],
            )[0]
        ]

    def search_threshold(
        self, search_embedded: Union[torch.Tensor, List[float]], output_field: str, threshold: float = 150, k: int = 50, offset: int = 0, collection_name: str = "images"
    ) -> List[Tuple[Any, float]]:
        """
        Searches for vectors within a specified distance threshold in a collection.

        Args:
            search_embedded (Union[torch.Tensor, List[float]]): The query vector.
            output_field (str): The field to return in results.
            threshold (float): Distance threshold. Defaults to 150.
            k (int): Maximum number of results to return. Defaults to 50.
            offset (int): Offset for paginated results. Defaults to 0.
            collection_name (str): Name of the collection. Defaults to "images".

        Returns:
            List[Tuple[Any, float]]: A list of tuples containing the output_field value and distance.
        """
        collection = Collection(name=collection_name)
        collection.load()

        if isinstance(search_embedded, torch.Tensor):
            search_embedded = search_embedded.tolist()

        param = {
            "metric_type": "L2",
            "offset": offset,
            "params": {"radius": threshold, "range_filter": 0.0},
        }

        results = collection.search(
            data=search_embedded,
            anns_field="embedding",
            param=param,
            limit=k,
            output_fields=[output_field],
        )

        return [
            (getattr(hit.entity, output_field), hit.distance)
            for hit in results[0]
        ]

    def query(
        self, collection_name: str, expr: str, output_fields: List[str], offset: int = 0, limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Queries a collection with a specific expression.

        Args:
            collection_name (str): Name of the collection to query.
            expr (str): Query expression.
            output_fields (List[str]): List of fields to include in the output.
            offset (int): Offset for paginated results. Defaults to 0.
            limit (int): Maximum number of results. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of query results.
        """
        col = Collection(collection_name)
        if not col:
            return None

        return col.query(
            expr=expr,
            offset=offset,
            limit=limit,
            output_fields=output_fields,
        )
