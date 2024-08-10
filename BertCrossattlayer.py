import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'#3
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import copy
import json
import logging
import math
import sys
from io import open
import torch
from torch import nn
import torch.nn.functional as F
import re
import numpy as np
import tensorflow as tf

try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                   Path.home() / '.pytorch_pretrained_bert'))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                              os.path.join(os.path.expanduser("~"), '.pytorch_pretrained_bert'))

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model



class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,

                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        # if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
        #                 and isinstance(vocab_size_or_config_json_file, unicode)):
        #     with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
        #         json_config = json.loads(reader.read())
        #     for key, value in json_config.items():
        #         self.__dict__[key] = value
        # elif isinstance(vocab_size_or_config_json_file, int):
        self.vocab_size = 30522
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        # else:
        #     raise ValueError("First argument must be either a vocabulary size (int)"
        #                      "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

BertLayerNorm = torch.nn.LayerNorm



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CMELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output



class CMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        # Number of layers
        self.num_l_layers = 12
        self.num_x_layers = 6 
        self.num_r_layers = 3
        print("Cross-Modality encoder with %d l_layers, %d x_layers, and %d r_layers." %
              (self.num_l_layers, self.num_x_layers, self.num_r_layers))

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        config.hidden_act = 'relu'
        self.x_layers = nn.ModuleList(
            [CMELayer(config) for _ in range(self.num_x_layers)]
        )
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        )

    def forward(self, lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.


        # Run language layers
        for layer_module in self.layer:
            lang_feats = layer_module(lang_feats, lang_attention_mask)

        # Run relational layers
        for layer_module in self.r_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask,
                                                  visn_feats, visn_attention_mask)

        return lang_feats, visn_feats

class CMDLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.cross_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.bert_inter = BertIntermediate(config)
        self.bert_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, q_input, kv_input, kv_mask):
        # Cross Attention
        att_output = self.cross_attention(q_input, kv_input, ctx_att_mask=kv_mask)
        return att_output

    def output_fc(self, input):
        # FC layers
        att_output = self.bert_inter(input)
        # Layer output
        output = self.bert_output(att_output, input)
        return output

    def forward(self, cm_feat, lang_feats, lang_attention_mask,
                      top_visn_feats, visn_feats,visn_attention_mask):
        
        img_cross_output = self.cross_att(cm_feat, top_visn_feats, visn_attention_mask)
        text_cross_output = self.cross_att(img_cross_output, visn_feats, lang_attention_mask)
        cross_output = self.output_fc(text_cross_output)

        return cross_output


class CMDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        # Number of layers
        self.num_s_layers = 6
        self.num_c_layers = 12
        print("Cross-Modality decoder with %d l_layers, %d x_layers." %
              (self.num_s_layers, self.num_c_layers))
        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_s_layers)]
        )
        config.hidden_act = 'relu'
        self.x_layers = nn.ModuleList(
            [CMDLayer(config) for _ in range(self.num_c_layers)]
        )


    def forward(self, cm_feat, lang_feats, lang_attention_mask,
                top_visn_feats, visn_feats, visn_attention_mask=None):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.

        # Run self layers
        for layer_module in self.layer:
            cm_feats = layer_module(cm_feat, None)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            out_feats = layer_module(cm_feats, lang_feats, lang_attention_mask,
                                                  top_visn_feats, visn_feats, visn_attention_mask)

        return out_feats

class MeanPooling(nn.Module):
    def __init__(self, args):
        super().__init__()
        
    def forward(self, decoder_out):
        
        last = decoder_out.transpose(1, 2)  
        out = F.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)          

        return out


class CNNPooling(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.conv1 =  nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0)  
        self.conv2 =  nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 =  nn.Conv1d(in_channels=768, out_channels=256, kernel_size=5, stride=1, padding=2)

        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True)
        self.maxpool = nn.AvgPool1d(23)

    def forward(self, features, mask):

        token_embeddings = features.transpose(1, -1)        
        out1 = self.conv1(token_embeddings)    
        out2 = self.conv2(token_embeddings)    
        out3 = self.conv3(token_embeddings)     
        vectors = [out1, out2, out3]
        out = torch.cat(vectors, 1).transpose(1, -1)    #

        out = self.ln(out)
        out = self.relu(out)

        out = out.transpose(1, 2)   
        out = F.avg_pool1d(out, kernel_size=out.shape[-1]).squeeze(-1)     

        return out












# class MUMCBASE(Transformer):
#     def __init__(self,args =None):
#         super().__init__(args=args )

#     self.temp = nn.Parameter(torch.ones([]) * 0.07)
#     self.register_buffer("image_queue", torch.randn(args.hidden_size, 65536))  # shape = [768, 65536]
#     self.register_buffer("text_queue", torch.randn(args.hidden_size, 65536))  # shape = [768, 65536]
#     self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
#     self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
#     self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

#     self.visual_encoder = self.trans
#     self.visual_encoder_m = self.trans_m
#     self.text_encoder = self.bert_embedding 
#     self.text_encoder = self.bert_embedding_m 

#     self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
#                                 [self.text_encoder, self.text_encoder_m],
#                                 ]
#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, image_feat, text_feat):
#         # gather keys before updating queue
#         if dist.is_available() and dist.is_initialized():
#             image_feats = concat_all_gather(image_feat)
#             text_feats = concat_all_gather(text_feat)
#         else:
#             image_feats = image_feat
#             text_feats = text_feat
#         # image_feats = concat_all_gather(image_feat)
#         # text_feats = concat_all_gather(text_feat)

#         batch_size = image_feats.shape[0]

#         ptr = int(self.queue_ptr)
#         assert self.queue_size % batch_size == 0  # for simplicity

#         # replace the keys at ptr (dequeue and enqueue)
#         self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
#         self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
#         ptr = (ptr + batch_size) % self.queue_size  # move pointer

#         self.queue_ptr[0] = ptr

#     @torch.no_grad()
#     def _momentum_update(self):
#         for model_pair in self.model_pairs:
#             for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
#                 param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)



#     def forward(self, image_feat, text_feat, image_feat_m, text_feat_m, alpha=0.4):
#         with torch.no_grad():
#             self.temp.clamp_(0.001, 0.5)
#         # get momentum features
#         with torch.no_grad():
#             self._momentum_update()
#             image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
#             text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
#             sim_i2t_m = image_feat_m @ text_feat_all / self.temp
#             sim_t2i_m = text_feat_m @ image_feat_all / self.temp

#             sim_targets = torch.zeros(sim_i2t_m.size()).to(image_feat.device)
#             sim_targets.fill_diagonal_(1)

#             sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
#             sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

#         sim_i2i = image_feat @ image_feat_all / self.temp
#         sim_t2t = text_feat @ text_feat_all / self.temp
#         sim_i2t = image_feat @ text_feat_all / self.temp
#         sim_t2i = text_feat @ image_feat_all / self.temp

#         loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
#         loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()
#         loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
#         loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

#         loss_ita = (loss_i2t + loss_t2i + 0.5 * (loss_i2i + loss_t2t)) / 2

#         self._dequeue_and_enqueue(image_feat_m, text_feat_m)
#         return loss_ita







