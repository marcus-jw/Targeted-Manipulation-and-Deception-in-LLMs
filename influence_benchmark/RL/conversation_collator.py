import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from transformers import DataCollatorForLanguageModeling


class DataCollatorMaskingStaticConversation(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
        ignore_first_n_messages (`int`, *optional*, defaults to `0`):
            The number of messages at the beginning of the conversation to ignore. This is useful for static messages
            which should not be trained on.
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(self.instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_idxs = []
            human_token_ids_idxs = []

            # Find all response and human instruction token indices # TODO: this seems pretty messy?
            for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                if self.response_token_ids == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist():
                    response_token_ids_idxs.append(idx + len(self.response_token_ids))

            if self.instruction_template is not None and self.instruction_token_ids is not None:
                for idx in np.where(batch["labels"][i] == self.instruction_token_ids[0])[0]:
                    if (
                        self.instruction_token_ids
                        == batch["labels"][i][idx : idx + len(self.instruction_token_ids)].tolist()
                    ):
                        human_token_ids_idxs.append(idx)

            if len(response_token_ids_idxs) == 0:
                warnings.warn(
                    "Could not find response key in the instance. This instance will be ignored in loss calculation."
                )
                batch["labels"][i, :] = self.ignore_index
                continue

            if self.instruction_template and len(human_token_ids_idxs) == 0:
                warnings.warn(
                    "Could not find instruction key in the instance. This instance will be ignored in loss calculation."
                )
                batch["labels"][i, :] = self.ignore_index
                continue

            # Ensure human_token_ids_idxs starts with 0 if necessary
            if self.instruction_template and human_token_ids_idxs[0] > response_token_ids_idxs[0]:
                human_token_ids_idxs = [0] + human_token_ids_idxs

            # Determine the start index for masking based on ignore_first_n_messages
            start_idx = 0
            num_messages_to_ignore = examples[i]["num_hardcoded_msgs"]  # type: ignore
            if self.instruction_template:
                if len(human_token_ids_idxs) > num_messages_to_ignore:
                    start_idx = human_token_ids_idxs[num_messages_to_ignore]
                else:
                    start_idx = len(batch["labels"][i])  # Ignore all if not enough messages
            else:
                if len(response_token_ids_idxs) > num_messages_to_ignore:
                    start_idx = response_token_ids_idxs[num_messages_to_ignore - 1]
                else:
                    start_idx = len(batch["labels"][i])  # Ignore all if not enough messages

            # Apply masking
            if self.instruction_template:
                for human_start, response_start in zip(
                    human_token_ids_idxs[num_messages_to_ignore:],
                    response_token_ids_idxs[num_messages_to_ignore:],
                ):
                    batch["labels"][i, human_start:response_start] = self.ignore_index
            else:  # TODO: what is the alternative to instruction template and why are we overriding start_idx?
                for response_start in response_token_ids_idxs[num_messages_to_ignore:]:
                    batch["labels"][i, start_idx:response_start] = self.ignore_index
                    start_idx = response_start

            # Mask everything before the start_idx
            batch["labels"][i, :start_idx] = self.ignore_index

        del batch["num_hardcoded_msgs"]
        return batch
