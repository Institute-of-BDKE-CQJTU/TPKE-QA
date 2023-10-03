import torch
from torch import nn
from torch.nn import Module, Linear, LayerNorm, CrossEntropyLoss
from torch.nn.parameter import Parameter

from transformers import BertPreTrainedModel, BertModel, RobertaModel
from transformers.modeling_bert import BertLMPredictionHead, ACT2FN
from transformers import PreTrainedTokenizer, BertTokenizer
from typing import Optional, Union, List, Dict, Tuple
from torch.nn import init
import numpy as np

class MultiHeadAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MultiHeadAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        residual = queries
        nk = keys.shape[1] # 64

        # print(queries.size())
        # print(self.fc_q(queries).size())
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            # print(attention_weights.size())
            # print(att.size())
            expand_attention_weights = attention_weights[:, None, None, None].expand(b_s, self.h, nq, nk)
            # att = att * expand_attention_weights
            att = att.masked_fill(~expand_attention_weights.bool(), 0.0)
        if attention_mask is not None:
            expand_attention_mask = attention_mask[:, None, None, :].expand(b_s, self.h, nq, nk)
            att = att.masked_fill(~expand_attention_mask.bool(), -np.inf)
        att = torch.softmax(att, -1)
        # print(att.size())
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return self.layer_norm(out+residual)

def gather_positions(input_tensor, positions):
    """
    :param input_tensor: shape [batch_size, seq_length, dim]
    :param positions: shape [batch_size, num_positions]
    :return: [batch_size, num_positions, dim]
    """
    _, _, dim = input_tensor.size()
    index = positions.unsqueeze(-1).repeat(1, 1, dim)  # [batch_size, num_positions, dim]
    gathered_output = torch.gather(input_tensor, dim=1, index=index)  # [batch_size, num_positions, dim]
    return gathered_output


class FullyConnectedLayer(Module):
    def __init__(self, input_dim, output_dim, hidden_act="gelu"):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense = Linear(self.input_dim, self.output_dim)
        self.act_fn = ACT2FN[hidden_act]
        self.LayerNorm = LayerNorm(self.output_dim)

    def forward(self, inputs):
        temp = self.dense(inputs)
        temp = self.act_fn(temp)
        temp = self.LayerNorm(temp)
        return temp


class QuestionAwareSpanSelectionHead(Module):
    def __init__(self, config):
        super().__init__()

        self.query_start_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.query_end_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.start_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.end_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)

        self.start_classifier = Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.end_classifier = Parameter(torch.Tensor(config.hidden_size, config.hidden_size))

    def forward(self, inputs, positions):
        gathered_reps = gather_positions(inputs, positions)

        query_start_reps = self.query_start_transform(gathered_reps)  # [batch_size, num_positions, dim]
        query_end_reps = self.query_end_transform(gathered_reps)  # [batch_size, num_positions, dim]
        start_reps = self.start_transform(inputs)  # [batch_size, seq_length, dim]
        end_reps = self.end_transform(inputs)  # [batch_size, seq_length, dim]

        temp = torch.matmul(query_start_reps, self.start_classifier)  # [batch_size, num_positions, dim]
        start_reps = start_reps.permute(0, 2, 1)  # [batch_size, dim, seq_length]
        start_logits = torch.matmul(temp, start_reps)

        temp = torch.matmul(query_end_reps, self.end_classifier)
        end_reps = end_reps.permute(0, 2, 1)
        end_logits = torch.matmul(temp, end_reps)

        return start_logits, end_logits


class ClassificationHead(Module):
    def __init__(self, config):
        super().__init__()
        self.span_predictions = QuestionAwareSpanSelectionHead(config)

    def forward(self, inputs, positions):
        return self.span_predictions(inputs, positions)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class ModelWithQASSHead(BertPreTrainedModel):
    def __init__(self, config, replace_mask_with_question_token=False, lambda_value=0.1, 
                 mask_id=103, question_token_id=104, sep_id=102, initialize_new_qass=True):
        super().__init__(config)
        self.encoder_name = config.model_type
        if "roberta" in self.encoder_name:
            self.roberta = RobertaModel(config)
        else:
            self.bert = BertModel(config)
        self.initialize_new_qass = initialize_new_qass
        self.cls = ClassificationHead(config) if not self.initialize_new_qass else None
        self.new_cls = ClassificationHead(config) if self.initialize_new_qass else None

        self.attention = MultiHeadAttention(d_model=768, d_k=96, d_v=96, h=8)

        self.lambda_value = lambda_value
        print(self.lambda_value)
        self.knowledge_cls = ClassificationHead(config)

        self.awloss = AutomaticWeightedLoss(2)

        self.replace_mask_with_question_token = replace_mask_with_question_token
        self.mask_id = mask_id
        self.question_token_id = question_token_id
        self.sep_id = sep_id

        self.init_weights()

    def get_cls(self):
        return self.cls if not self.initialize_new_qass else self.new_cls

    def get_encoder(self):
        if "roberta" in self.encoder_name:
            return self.roberta
        return self.bert

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, masked_positions = None,
                knowledge_input_ids_total = None, knowledge_mask_total=None, knowledge_attention_mask_total = None, knowledge_token_type_ids_total = None, knowledge_answer_start_total = None, knowledge_answer_end_total = None,
                knowledge_weights=None,
                start_positions=None, end_positions=None):

        if attention_mask is not None:
            attention_mask[input_ids == self.sep_id] = 0
        if self.replace_mask_with_question_token:
            input_ids = input_ids.clone()
            input_ids[input_ids == self.mask_id] = self.question_token_id
            

        mask_positions_were_none = False
        if masked_positions is None:
            masked_position_for_each_example = torch.argmax((input_ids == self.question_token_id).int(), dim=-1)
            masked_positions = masked_position_for_each_example.unsqueeze(-1)
            mask_positions_were_none = True
            
        encoder = self.get_encoder()
        knowledge_sequence_output = None
        if knowledge_input_ids_total is not None:
            knowledge_input_ids_total = knowledge_input_ids_total.clone()
            knowledge_input_ids_total[knowledge_input_ids_total == self.mask_id] = self.question_token_id
            knowledge_masked_positions_for_each_example = torch.argmax((knowledge_input_ids_total == self.question_token_id).int(), dim=-1)
            knowledge_masked_positions = knowledge_masked_positions_for_each_example.unsqueeze(-1)
            
            knowledge_outputs = encoder(input_ids=knowledge_input_ids_total, attention_mask=knowledge_attention_mask_total, token_type_ids=knowledge_token_type_ids_total)
            knowledge_sequence_output = knowledge_outputs[0]  # [batch_size, max_length, dim]
            if knowledge_answer_start_total is not None:
                knowledge_start_logits, knowledge_end_logits = self.new_cls(knowledge_sequence_output, knowledge_masked_positions)
                knowledge_start_logits, knowledge_end_logits = knowledge_start_logits.squeeze(1), knowledge_end_logits.squeeze(1)
                knowledge_start_logits = knowledge_start_logits + (1 - knowledge_attention_mask_total) * -10000.0
                knowledge_end_logits = knowledge_end_logits + (1 - knowledge_attention_mask_total) * -10000.0
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                knowledge_start_loss = loss_fct(knowledge_start_logits, knowledge_answer_start_total.long())
                knowledge_end_loss = loss_fct(knowledge_end_logits, knowledge_answer_end_total.long())
                
                knowledge_loss = (knowledge_start_loss+knowledge_end_loss)/2
        
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]  # [batch_size, max_length, dim]
        if knowledge_input_ids_total is not None:
            sequence_output_attention = self.attention(sequence_output, knowledge_sequence_output, knowledge_sequence_output, attention_mask=knowledge_mask_total, attention_weights=knowledge_weights)
        
        
        cls = self.get_cls()
        start_logits, end_logits = cls(sequence_output, masked_positions)
        if knowledge_input_ids_total is not None:
            start_logits_attention, end_logits_attention = cls(sequence_output_attention, masked_positions)

        if mask_positions_were_none:
            start_logits, end_logits = start_logits.squeeze(1), end_logits.squeeze(1)
            if knowledge_input_ids_total is not None:
                start_logits_attention, end_logits_attention = start_logits_attention.squeeze(1), end_logits_attention.squeeze(1)

        if attention_mask is not None:
            start_logits = start_logits + (1 - attention_mask) * -10000.0
            end_logits = end_logits + (1 - attention_mask) * -10000.0
            if knowledge_input_ids_total is not None:
                start_logits_attention = start_logits_attention + (1 - attention_mask) * -10000.0
                end_logits_attention = end_logits_attention + (1 - attention_mask) * -10000.0

        outputs = outputs[2:]
        if knowledge_input_ids_total is not None:
            outputs = (start_logits_attention, end_logits_attention, ) + outputs
        else:
            outputs = (start_logits, end_logits, ) + outputs

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            # print(ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            start_loss = loss_fct(start_logits, start_positions.long())
            end_loss = loss_fct(end_logits, end_positions.long())

            total_loss = (start_loss + end_loss) / 2
            
            if knowledge_input_ids_total is not None:
                # ignored_index = start_logits_attention.size(1)

                loss_fct = CrossEntropyLoss(ignore_index=-100)
                start_attention_loss = loss_fct(start_logits_attention, start_positions.long())
                end_attention_loss = loss_fct(end_logits_attention, end_positions.long())

                total_loss = total_loss + (start_attention_loss + end_attention_loss) / 2
                
                total_loss = total_loss + self.lambda_value * knowledge_loss

            outputs = (total_loss,) + outputs

        return outputs


