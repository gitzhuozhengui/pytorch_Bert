# -*- coding:utf-8 -*-
# @Author   : zzg
# @Time     : 2022/12/15
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from addr_config import bert_path


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc_1 = nn.Linear(768, 32)  # 768 -> 32
        self.fc_2 = nn.Linear(32, 2)  # 32 -> 2
        self.quant = torch.quantization.QuantStub()  # QuantStub: 转换张量从浮点到量化
        self.dequant = torch.quantization.DeQuantStub()  # DeQuantStub: 将量化张量转换为浮点

    def forward(self, x):
        # x = [i.to('cpu') for i in x]
        # x = [self.quant(i) for i in x]
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(
            context,
            token_type_ids=types,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )  # 控制是否输出所有encoder层的结果
        out = self.fc_1(pooled)  # 得到2分类
        out = self.fc_2(out)  # 得到2分类
        # out = self.dequant(out)  # 手动指定张量: 从量化转换到浮点
        return out

