import copy
import logging
from typing import Optional, Tuple, Union, List
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers import T5Config, T5ForConditionalGeneration
from generation_utils_vision_t5 import GenerationMixin_VisionT5
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, BaseModelOutput, \
    ModelOutput


# logger = logging.get_logger(__name__)


class Encoder(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        # ---- Modified ----#
        # add visual features (without position features)
        visual_feat_embedding = [nn.Linear(config.feat_dim, config.d_model)]
        self.visual_feat_embedding = nn.Sequential(*visual_feat_embedding)
        # ------------------#

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0))
             for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        self.vis_embeds = None

    def forward(
            self,
            input_ids=None,
            attention_mask=None,

            vis_feats=None,
            vis_attention_mask=None,

            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # ---- Modified ----#
        # add vis_feats and vis_attention_mask
        # ------------------#

        # Model parallel
        # ---- Modified ----#
        # delete

        # if self.model_parallel:
        #     torch.cuda.set_device(self.first_device)
        #     self.embed_tokens = self.embed_tokens.to(self.first_device)
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if input_ids is not None and inputs_embeds is not None:
        #     err_msg_prefix = "decoder_" if self.is_decoder else ""
        #     raise ValueError(
        #         f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
        #     )
        # elif input_ids is not None:
        #     input_shape = input_ids.size()
        #     input_ids = input_ids.view(-1, input_shape[-1])
        # elif inputs_embeds is not None:
        #     input_shape = inputs_embeds.size()[:-1]
        # else:
        #     err_msg_prefix = "decoder_" if self.is_decoder else ""
        #     raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")
        # ------------------#

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
        # ---- Modified ----#
        # add
        vis_embeds = self.visual_feat_embedding(vis_feats)
        inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)
        # ------------------#

        batch_size, text_seq_length = inputs_embeds.size()[:-1]
        vis_seq_length = vis_embeds.size(1)
        self.vis_embeds = vis_embeds

        seq_length = text_seq_length + vis_seq_length
        input_shape = (batch_size, text_seq_length + vis_seq_length)

        # required mask seq length can be calculated via length of past
        mask_text_seq_length = past_key_values[0][0].shape[2] + \
                               text_seq_length if past_key_values is not None else text_seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_text_seq_length).to(inputs_embeds.device)

        # ---- Modified ----#
        # add vis_attention_mask
        if vis_attention_mask is None:
            # vis_attention_mask = torch.ones(batch_size, vis_seq_length).to(inputs_embeds.device)
            # new_ones returns same tensor.dtype and device
            vis_attention_mask = attention_mask.new_ones(
                batch_size, vis_seq_length)

        attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)
        # ------------------#

        # ---- Modified ----#
        # delete
        # if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        #     encoder_seq_length = encoder_hidden_states.shape[1]
        #     encoder_attention_mask = torch.ones(
        #         batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
        #     )
        # ------------------#

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # ---- Modified ----#
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.device)
        # ------------------#

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, text_seq_length, text_seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        # ---- Modified ----#
        # add
        # if self.config.num_layers > 0:
        #     assert self.block[0].layer[0].SelfAttention.has_relative_attention_bias
        #     text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(
        #         text_seq_length, text_seq_length)
        #     num_heads = text_position_bias.size(1)
        #     position_bias = text_position_bias.new_zeros(
        #         1, num_heads, seq_length, seq_length
        #     )
        #     position_bias[:, :, :text_seq_length,:text_seq_length] = text_position_bias
        #     breakpoint()
        #     position_bias = position_bias + extended_attention_mask

        # ------------------#

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            # Model parallel
            # ---- Modified ----#
            # delete
            # if self.model_parallel:
            #     torch.cuda.set_device(hidden_states.device)
            #     # Ensure that attention_mask is always on the same device as hidden_states
            #     if attention_mask is not None:
            #         attention_mask = attention_mask.to(hidden_states.device)
            #     if position_bias is not None:
            #         position_bias = position_bias.to(hidden_states.device)
            #     if encoder_hidden_states is not None:
            #         encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            #     if encoder_extended_attention_mask is not None:
            #         encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            #     if encoder_decoder_position_bias is not None:
            #         encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            #     if layer_head_mask is not None:
            #         layer_head_mask = layer_head_mask.to(hidden_states.device)
            #     if cross_attn_layer_head_mask is not None:
            #         cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)
            # ------------------#

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    # logger.warning(
                    #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    # )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                # ---- Modified ----#
                # layer_outputs = layer_module(
                #     hidden_states,
                #     attention_mask=extended_attention_mask,
                #     position_bias=position_bias,
                #     encoder_hidden_states=encoder_hidden_states,
                #     encoder_attention_mask=encoder_extended_attention_mask,
                #     encoder_decoder_position_bias=encoder_decoder_position_bias,
                #     layer_head_mask=layer_head_mask,
                #     cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                #     past_key_value=past_key_value,
                #     use_cache=use_cache,
                #     output_attentions=output_attentions,
                # )
                # ------------------#
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=None,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    head_mask=layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with: hidden-states, key-value-states, (self-attention position bias),
            # (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + \
                                           (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + \
                                           (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            # if self.model_parallel:
            #     for k, v in self.device_map.items():
            #         if i == v[-1] and "cuda:" + str(k) != self.last_device:
            #             hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class VisionT5(GenerationMixin_VisionT5, T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)

        self.config = config
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # ---- Modified ----#
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = Encoder(encoder_config, self.shared)
        # ------------------#

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # NOTE
        self.classifier = nn.Linear(config.d_model, config.pos_dim, bias=False)

        # Initialize weights and apply final processing
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.vinvl_region_number = config.vinvl_region_number

    def get_input_embeddings(self):
        return super().get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return super().set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        return super().set_output_embeddings(new_embeddings)

    def get_encoder(self):
        return super().get_encoder()

    def get_decoder(self):
        return super().get_decoder()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,

            vis_feats=None,
            vis_attention_mask=None,
            img_label=None,

            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            # cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,

                vis_feats=vis_feats,
                vis_attention_mask=vis_attention_mask,

                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            # assert labels is not None, "Decoder should not use cached key states when training." # TODO
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(
                dtype=hidden_states.dtype, device=hidden_states.device)

        if vis_attention_mask is None:
            batch_size, text_seq_length = attention_mask.size()
            vis_seq_length = encoder_outputs[0].size(1) - text_seq_length
            vis_attention_mask = attention_mask.new_ones(batch_size, vis_seq_length)
        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,

            head_mask=decoder_head_mask,
            # cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        # _sequence_output = decoder_outputs[0] 

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        vis_similarities = torch.matmul(sequence_output, self.encoder.vis_embeds.transpose(1, 2))

        loss = None
        klloss = None

        if labels is not None:
            # CEloss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            # KLloss
            kl_loss = KLDivLoss(reduction="batchmean")
            mask_for_classifier_index = []

            for batch_i, label in enumerate(labels):
                _list_total = 0
                this_batch = []
                flag = True
                for ll in label:
                    if ll == -100:
                        this_batch.extend([False] * _list_total)
                        this_batch.extend([False] * (label.size(0) - len(this_batch)))
                        _list_total = 0
                        break
                    if ll == 59:
                        this_batch.append(False)
                        this_batch.extend([False] * _list_total)
                        _list_total = 0
                        flag = False
                        continue
                    if ll not in [16, 8, 1023]:  # the id of "in the image"
                        this_batch.append(False)
                        this_batch.extend([False] * _list_total)
                        _list_total = 0
                    elif ll == [16, 8, 1023][_list_total]:
                        if not flag:
                            this_batch.append(False)
                            this_batch.extend([False] * _list_total)
                            _list_total = 0
                            flag = True
                            continue
                        _list_total += 1
                        if _list_total == 3:
                            this_batch.extend([True, True, True])
                            _list_total = 0

                this_batch.extend([False] * _list_total)
                mask_for_classifier_index.append(this_batch)

            mask_for_classifier_index = torch.tensor(mask_for_classifier_index)
            # mask_for_classifier_index = mask_for_classifier_index.unsqueeze(-1).expand(-1,-1,hidden_states.size(-1)).to(hidden_states.device)

            vis_similarities = vis_similarities[mask_for_classifier_index]
            # vis_similarities = F.softmax(vis_similarities, dim=-1)
            mask_img_label = []
            for batch_i, label in enumerate(img_label):
                this_batch = []
                for entity in label:
                    if sum(entity) == 0:
                        this_batch.append(False)
                    # elif sum(entity) == -100:   # object detection fault
                    #     this_batch.append(True)
                    else:
                        this_batch.append(True)

                mask_img_label.append(this_batch)

            mask_img_label = torch.tensor(mask_img_label)
            img_label = img_label[mask_img_label]
            # img_label = torch.where(img_label < 0, torch.tensor(0.0).to(vis_similarities.device), img_label)
            if img_label.size(0) == 0:
                klloss = torch.tensor(0.0).to(vis_similarities.device)
            else:
                klloss = kl_loss(
                    input=F.log_softmax(vis_similarities.view(-1, 3, self.vinvl_region_number).mean(dim=1), dim=1),
                    # '3' means len("in the image")
                    target=img_label)
            loss += klloss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput_VisionT5(
            loss=loss,
            logits=lm_logits,
            vis_similarities=vis_similarities,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            # cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            # "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

        if 'vis_attention_mask' in kwargs:
            output['vis_attention_mask'] = kwargs['vis_attention_mask']

        return output

    def _expand_inputs_for_generation(
            input_ids,
            expand_size=1,
            is_encoder_decoder=False,
            attention_mask=None,
            encoder_outputs=None,
            **model_kwargs,
    ):

        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        # ---- Modified ----#
        if model_kwargs.get('vis_attention_mask', None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(0, expanded_return_idx)
        # ------------------#

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs


class Seq2SeqLMOutput_VisionT5(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    vis_similarities: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
