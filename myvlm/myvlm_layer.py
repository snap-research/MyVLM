import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, Union, List

from concept_embedding_training.data_utils import cosine_distance
from myvlm.common import MyVLMLayerMode


class MyVLMLayer(torch.nn.Module):

    def __init__(self,
                 layer: torch.nn.Module,
                 embedding_dim: int,
                 threshold: float,
                 torch_dtype: torch.dtype = torch.bfloat16,
                 device: str = 'cuda'):
        super().__init__()
        self.layer = layer
        self.threshold = threshold
        self.torch_dtype = torch_dtype
        self.device = device
        self.value_shape = embedding_dim
        self.keys, self.values = None, None
        self.key_idx_to_value_idx = {}
        self.mode = MyVLMLayerMode.TRAIN

    def forward(self, *args) -> torch.Tensor:
        # Run layer forward and save what it would have returned for this instance
        hidden_state = args[0]
        concept_signal = args[1]

        layer_out = self.layer(hidden_state)

        # If we're not training and we have no learned keys yet, just return the original layer output
        if not self.training and 'keys' not in self.__dict__:
            return layer_out

        # Let's extract the query from the input
        query = self._get_query(concept_signal)
        if query is None:
            return layer_out

        # Initialize a new concept embedding if this is the first iteration of training
        new_embedding = None
        if self.iter == 0:
            new_embedding = self._init_new_concept_embedding()

        # If no key/value exists, initialize them
        if 'keys' not in self.__dict__ or self.keys is None:
            self._init_keys_and_values(concept_signal=concept_signal,
                                       query=query,
                                       new_embedding=new_embedding)

        # Main logic: try seeing if we need to add the concept embedding to the layer output
        extended_layer_out = self._possibly_add_concept_embedding(layer_out=layer_out,
                                                                  query=query,
                                                                  concept_signal=concept_signal)
        return extended_layer_out

    def _possibly_add_concept_embedding(self,
                                        layer_out: torch.Tensor,
                                        query: torch.Tensor,
                                        concept_signal: Union[torch.Tensor, List]) -> torch.Tensor:
        self.keys = self.keys.to(self.device)
        # Case 1: Working with objects
        if type(concept_signal[0]) == dict:
            return self._possibly_add_object_embedding(layer_out=layer_out,
                                                       query=query,
                                                       concept_signal=concept_signal)
        # Case 2: Working with people where multiple people were detected in the image
        elif type(concept_signal) == torch.Tensor and concept_signal.shape[1] > 1:
            return self._possibly_add_multi_face_embedding(layer_out=layer_out,
                                                           concept_signal=concept_signal)
        # Case 3: Working with people where only a single face was detected
        else:
            return self._possibly_add_single_face_embedding(layer_out=layer_out,
                                                            query=query,
                                                            concept_signal=concept_signal)

    def _possibly_add_object_embedding(self,
                                       layer_out: torch.Tensor,
                                       query: List,
                                       concept_signal: Union[torch.Tensor, List]) -> torch.Tensor:
        """ This handles the logic for adding object concept embeddings to the layer output. """
        extended_layer_out = []
        dists = self._compute_distances(concept_signal=concept_signal, query=query)
        # Iterate over all the images that we got in the batch
        for sample_idx, q in enumerate(concept_signal):
            sample_dists = dists[sample_idx]
            sample_out = layer_out[sample_idx]
            previously_added_concept_idxs = set()  # Store concepts that were added so we don't add them twice
            for concept_idx, dist in sample_dists.items():
                if dist <= self.threshold:
                    if self.mode == MyVLMLayerMode.INFERENCE:
                        print(f"Adding concept: {concept_idx}. "
                              f"Distance: {dist.item():0.3f} | "
                              f"Threshold: {self.threshold}")
                    # Normalize the concept embedding before we add it to the layer output
                    value_to_add = F.normalize(self.values[concept_idx], dim=-1, p=2)
                    # Concatenate the concept embedding to the layer output
                    sample_out = torch.vstack([sample_out, value_to_add.unsqueeze(0)])
                    previously_added_concept_idxs.add(concept_idx)
                else:
                    if self.mode == MyVLMLayerMode.INFERENCE:
                        print(f"Not adding concept: {concept_idx}. "
                              f"Distance: {dist.item():0.3f} |"
                              f"Threshold: {self.threshold}")
            extended_layer_out.append(sample_out)
        return torch.stack(extended_layer_out, dim=0).to(dtype=self.torch_dtype)

    def _possibly_add_multi_face_embedding(self,
                                           layer_out: torch.Tensor,
                                           concept_signal: Union[torch.Tensor, List]) -> torch.Tensor:
        """ This handles the logic for adding the concept embedding when there may be multiple faces in the image. """
        extended_layer_out = []
        for sample_idx, q in enumerate(concept_signal):
            dists = self._compute_distances(concept_signal=concept_signal, query=q.to(self.device))
            smallest_dist, chosen_key = dists.min(0)
            smallest_dist = smallest_dist.view(-1, 1).to(self.device)
            value_idxs = [self.key_idx_to_value_idx[k.item()] for k in chosen_key]
            chosen_value = self.values[value_idxs]
            sample_out = layer_out[sample_idx]
            previously_added_concept_idxs = set()  # Store concepts that were added so we don't add them twice
            for concept_idx in range(len(value_idxs)):
                if value_idxs[concept_idx] in previously_added_concept_idxs:
                    print(f"Concept {value_idxs[concept_idx]} was already added to the layer output. Skipping.")
                    continue
                if smallest_dist[concept_idx] <= self.threshold:
                    if self.mode == MyVLMLayerMode.INFERENCE:
                        print(f"Adding concept: {concept_idx}. "
                              f"Distance: {smallest_dist[concept_idx].item():0.2f} | "
                              f"Threshold: {self.threshold}")
                    # Normalize the concept embedding before we add it to the layer output
                    value_to_add = F.normalize(chosen_value[concept_idx], dim=-1, p=2)
                    # Concatenate the concept embedding to the layer output
                    sample_out = torch.vstack([sample_out, value_to_add.unsqueeze(0)])
                    previously_added_concept_idxs.add(value_idxs[concept_idx])
            extended_layer_out.append(sample_out)
        # Stack the new results back into a batch
        if len(extended_layer_out) > 0:
            layer_out = torch.stack(extended_layer_out, dim=0).to(dtype=self.torch_dtype)
        return layer_out

    def _possibly_add_single_face_embedding(self,
                                            layer_out: torch.Tensor,
                                            query: torch.Tensor,
                                            concept_signal: Union[torch.Tensor, List]) -> torch.Tensor:
        """ This handles the logic for adding the concept embedding when all images in the batch have a single face. """
        dists = self._compute_distances(concept_signal=concept_signal, query=query)
        smallest_dist, chosen_key = dists.min(0)
        smallest_dist = smallest_dist.view(-1, 1)
        # Map the chosen key to the index of the actual value (since we have multiple keys pointing to the same value)
        value_idxs = [self.key_idx_to_value_idx[k.item()] for k in chosen_key]
        chosen_value = self.values[value_idxs]
        if any(smallest_dist <= self.threshold):
            if self.mode == MyVLMLayerMode.INFERENCE:
                for idx, dist in enumerate(smallest_dist):
                    if dist <= self.threshold:
                        print(f"Sample {idx}: adding concept {value_idxs[idx]}. "
                              f"Distance: {dist.item():0.2f} | Threshold: {self.threshold}")
                    else:
                        print(f"Sample {idx}: not adding concepts. "
                              f"Distance: {dist.item():0.2f} | Threshold: {self.threshold}")

            # Normalize the concept embedding before we add it to the layer output
            value_to_add = F.normalize(chosen_value, dim=-1, p=2)
            # Concatenate the concept embedding to the layer output
            return torch.cat([layer_out, value_to_add.unsqueeze(1)], dim=1).to(dtype=layer_out.dtype)
        else:
            # No close concept was found for all images, return the original network output
            print(f"No close concept found. "
                  f"Min Distance: {min(smallest_dist).item():0.2f} | Threshold: {self.threshold}.")
            return layer_out

    def _get_query(self, concept_signal: Union[torch.Tensor, List]) -> Union[torch.Tensor, List]:
        # Case 1: Working with face embeddings
        if concept_signal is not None and type(concept_signal) == torch.Tensor:
            # If we have multiple faces (will only happen at inference), we'll need to handle this a bit differently
            if concept_signal.shape[1] > 1:
                query = [q.unsqueeze(0) for q in concept_signal[0]]
            # Only one face in the image, so just take it
            else:
                query = concept_signal[:, 0, :]
        # Case 2: we're working with classifier logits
        elif concept_signal is not None and type(concept_signal[0]) == dict:
            query = concept_signal
        # Case 3: no embeddings/logits were extracted, so we'll just return the original layer output
        else:
            query = None
        return query

    def _init_new_concept_embedding(self) -> torch.nn.Parameter:
        """ Initialize a trainable embedding for a new concept. """
        new_embedding = torch.rand(1, self.value_shape, requires_grad=True, device=self.device)
        new_embedding = new_embedding / new_embedding.norm(dim=-1, keepdim=True)
        new_embedding = torch.nn.Parameter(new_embedding.to(dtype=self.torch_dtype))
        return new_embedding

    def _init_keys_and_values(self,
                              concept_signal: Union[torch.Tensor, List],
                              query: torch.Tensor,
                              new_embedding: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the concept signal and the query, we'll initialize the keys and values for the concept embedding we're
        going to train.
        """
        if type(concept_signal[0]) == dict:
            # If we have logits, we'll just use the first one as the query. It's a bit less important to be honest.
            query = torch.cat([list(q.values())[0] for q in query])

        self.keys, self.values = self._init_key_value(new_key=query[:1], new_value=new_embedding)
        n_keys = len(self.keys)
        # Store a mapping from the key index to its value index
        self.key_idx_to_value_idx[len(self.key_idx_to_value_idx)] = n_keys - 1
        # For the remaining queries, we don't need to create a new value. All we need to do is add the key and map it
        # to the existing value we just added above
        for idx, q in enumerate(query[1:]):
            self.keys, self.values = self._add_key(new_key=q.unsqueeze(0))
            self.key_idx_to_value_idx[idx + n_keys] = n_keys - 1

    def _init_key_value(self, new_key: torch.Tensor, new_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Initializes the keys and values for holding our concept embedding. """
        keys = new_key.clone().detach()
        values = new_value
        return keys, values

    def _add_key(self,
                 new_key: torch.Tensor,
                 new_value: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adds a new key and value to the list of keys and values. Here, a key represents a specific image of a concept
        while the value represents the embedding of that concept. Note that multiple keys can be mapped to the same
        value (e.g., in our case, we use 4 images to define a single concept embedding).
        """
        # Add new key to list of keys
        keys = torch.vstack([self.keys, new_key.clone().detach()])
        # Add new value to list of values if specified
        values = self.values
        if new_value is not None:
            values = torch.nn.Parameter(torch.vstack([self.values, new_value]), requires_grad=True)
        return keys, values

    def _compute_distances(self, concept_signal: torch.Tensor, query: torch.Tensor) -> Union[torch.Tensor, List]:
        """ Computes the distance/probability for each query to contain the target concept. """
        if type(concept_signal) == list and type(concept_signal[0]) == dict:
            # This case handles the linear classifier probabilities for object concepts.
            # Here, we get distance to having probability of 1 for containing the concept
            dists = []
            for sample_probas in concept_signal:
                dists.append({k: 1 - v[0][1] for k, v in sample_probas.items()})
        else:
            # This case handles the face embeddings for people concepts
            dists = torch.stack([cosine_distance(query, key).view(-1, 1) for key in self.keys]).view(-1, len(query))

        return dists
