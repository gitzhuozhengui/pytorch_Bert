# -*- coding:utf-8 -*-
# @Author   : zzg
# @Time     : 2020/7/30

from logic.common.common import local_log
from server.core.model_manager import method_register
from server.module.model_base.method_train_model import MethodBaseTrainModel
from server.module.model_base.method_data_collect import MethodBaseDataCollect
from logic.common.mysql_pool import MysqlDbTools
import numpy as np
import time
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from addr_config import bert_path, model_addr
from server.module.model_chat.bert_model import Model
from pytorch_pretrained_bert import BertTokenizer

class TrainModel_Test(MethodBaseTrainModel):
  
    def __init__(self,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train_model(self, **kwargs):
        data_list = kewrage.get("data_list")
        train_df, test_df, dev_df = data_list[:]
        tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器

        max_len = 32
        batch_size = 64
        dev_loader = get_dataloader(
            dev_df, tokenizer, max_len, batch_size, data_type="dev"
        )
        test_loader = get_dataloader(
            test_df, tokenizer, max_len, batch_size, data_type="test"
        )
        train_loader = get_dataloader(
            train_df, tokenizer, max_len, batch_size, data_type="train"
        )
        
        model = Model().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # 简单起见，可用这一行代码完事
        best_acc = 0.0
        NUM_EPOCHS = 3
        PATH = model_addr + "roberta_model_test.pth"  # 定义模型保存路径
        for epoch in range(1, NUM_EPOCHS + 1):  # 3个epoch
            train(model, DEVICE, train_loader, optimizer, epoch)
            acc = test(model, DEVICE, test_loader)
            if best_acc < acc:
                best_acc = acc
                torch.save(model.state_dict(), PATH)  # 保存最优模型
            print("nTest acc is: {:.4f}, best acc is {:.4f}n".format(acc, best_acc))
        model.load_state_dict(torch.load(PATH))
        acc = test(model, DEVICE, test_loader)
        print("nTest acc is: {:.4f}".format(acc))

      def train(self, model, device, train_loader, optimizer, epoch):  # 训练模型
        model.train()
        for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            y_pred = model([x1, x2, x3])  # 得到预测结果
            model.zero_grad()  # 梯度清零
            loss = F.cross_entropy(y_pred, y.squeeze())  # 得到loss
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:  # 打印loss
                print(
                    "Train Epoch: {} [{}/{} ({:.2f}%)]tLoss: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * len(x1),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )  # 记得为loss.item()


    def test(model, device, test_loader):  # 测试模型, 得到测试集评估结果
        model.eval()
        test_loss = 0.0
        acc = 0
        for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            with torch.no_grad():
                y_ = model([x1, x2, x3])
            test_loss += F.cross_entropy(y_, y.squeeze())
            pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
            acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
        test_loss /= len(test_loader)
        print(
            "nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                test_loss,
                acc,
                len(test_loader.dataset),
                100.0 * acc / len(test_loader.dataset),
            )
        )
        return acc / len(test_loader.dataset)


    def get_dataloader(data, tokenizer, max_len, BATCH_SIZE, data_type="train"):
        input_ids, input_types, input_masks, label = [], [], [], []
        for res in data.values.tolist():
            x1, y = res[:]
            x1 = tokenizer.tokenize(x1)
            tokens = ["[CLS]"] + x1 + ["[SEP]"]
            # 得到input_id, seg_id, att_mask
            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * (len(ids))
            masks = [1] * len(ids)
            # 短则补齐，长则切断
            if len(ids) < max_len:
                types = types + [1] * (max_len - len(ids))  # mask部分 segment置为1
                masks = masks + [0] * (max_len - len(ids))
                ids = ids + [0] * (max_len - len(ids))
            else:
                types = types[:max_len]
                masks = masks[:max_len]
                ids = ids[:max_len]
            input_ids.append(ids)
            input_types.append(types)
            input_masks.append(masks)
            # print(len(ids), len(masks), len(types))
            assert len(ids) == len(masks) == len(types) == max_len
            label.append([int(y)])

        tensor_data = TensorDataset(
            torch.LongTensor(input_ids),
            torch.LongTensor(input_types),
            torch.LongTensor(input_masks),
            torch.LongTensor(label),
        )
        if data_type == "train":
            data_sampler = RandomSampler(tensor_data)
        else:
            data_sampler = SequentialSampler(tensor_data)
        data_loader = DataLoader(tensor_data, sampler=data_sampler, batch_size=BATCH_SIZE)
        return data_loader


    def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None):
        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import unique_labels

        if not title:
            if normalize:
                title = "Normalized confusion matrix"
            else:
                title = "Confusion matrix, without normalization"

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        print(cm)
