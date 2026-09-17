from torch import cat,squeeze,unsqueeze,sum
from torch.nn import Embedding,Module,Sigmoid,Tanh,Dropout,Linear
from torch.autograd import Variable
import torch
import torch.nn as nn

from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings

@MODEL_REGISTRY.register("dimkt")
class DIMKT(Module):
    def __init__(self,num_q,num_c,dropout=0.2,emb_size=128,batch_size=64,num_steps=200,difficult_levels=100,emb_type="qid",emb_path="", **kwargs):
        super().__init__()
        self.model_name = "dimkt"
        self.num_q = num_q  
        self.num_c = num_c
        self.emb_type = emb_type
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.difficult_levels = difficult_levels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.dropout = Dropout(dropout)
        
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.knowledge = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.emb_size)), requires_grad=True)

        self.q_emb = Embedding(self.num_q+1,self.emb_size,padding_idx=0)
        self.c_emb = Embedding(self.num_c+1,self.emb_size,padding_idx=0)
        self.sd_emb = Embedding(self.difficult_levels+2,self.emb_size,padding_idx=0)
        self.qd_emb = Embedding(self.difficult_levels+2,self.emb_size,padding_idx=0)
        self.a_emb = Embedding(2,self.emb_size)
        
        self.linear_1 = Linear(4*self.emb_size,self.emb_size)
        self.linear_2 = Linear(1*self.emb_size,self.emb_size)
        self.linear_3 = Linear(1*self.emb_size,self.emb_size)
        self.linear_4 = Linear(2*self.emb_size,self.emb_size)
        self.linear_5 = Linear(2*self.emb_size,self.emb_size)
        self.linear_6 = Linear(4*self.emb_size,self.emb_size)
        
            
    def forward(self,q,c,sd,qd,a,qshft,cshft,sdshft,qdshft):
        device = q.device
        if self.batch_size != len(q):
            self.batch_size = len(q)
        q_emb = self.q_emb(Variable(q))
        # Identity on [B,T]; on [B,T,K] mean-pools the question's KCs with -1
        # padding masked, as pykt's QueEmb.get_avg_skill_emb does.  q/sd/qd/a
        # are indexed by question, difficulty and response, so they stay put.
        c_emb = pool_concept_embeddings(self.c_emb, c, self.num_c)
        sd_emb = self.sd_emb(Variable(sd))
        qd_emb = self.qd_emb(Variable(qd))
        a_emb = self.a_emb(Variable(a))

        target_q = self.q_emb(Variable(qshft))
        target_c = pool_concept_embeddings(self.c_emb, cshft, self.num_c)
        target_sd = self.sd_emb(Variable(sdshft))
        target_qd = self.qd_emb(Variable(qdshft))
       
        input_data = cat((q_emb,c_emb,sd_emb,qd_emb),-1)
        input_data = self.linear_1(input_data)

        target_data = cat((target_q,target_c,target_sd,target_qd),-1)
        target_data = self.linear_1(target_data)

        
        shape = list(sd_emb.shape)
        padd = torch.zeros(shape[0],1,shape[2],device=device)
        sd_emb = cat((padd,sd_emb),1)
        slice_sd_embedding = sd_emb.split(1,dim=1)

        shape = list(a_emb.shape)
        padd = torch.zeros(shape[0],1,shape[2],device=device)
        a_emb = cat((padd,a_emb),1)
        slice_a_embedding = a_emb.split(1,dim=1)

        shape = list(input_data.shape)
        padd = torch.zeros(shape[0],1,shape[2],device=device)
        input_data = cat((padd,input_data),1)
        slice_input_data = input_data.split(1,dim=1)

        qd_emb = cat((padd,qd_emb),1)
        slice_qd_embedding = qd_emb.split(1,dim=1)
        
        k = self.knowledge.repeat(self.batch_size,1).to(device)
        
        h = list()
        seqlen = q.size(1)
        for i in range(1,seqlen+1):
            
            sd_1 = squeeze(slice_sd_embedding[i],1)
            a_1 = squeeze(slice_a_embedding[i],1)
            qd_1 = squeeze(slice_qd_embedding[i],1)
            input_data_1 = squeeze(slice_input_data[i],1)
            
            qq = k-input_data_1

            gates_SDF = self.linear_2(qq)
            gates_SDF = self.sigmoid(gates_SDF)
            SDFt = self.linear_3(qq)
            SDFt = self.tanh(SDFt) 
            SDFt = self.dropout(SDFt)

            SDFt = gates_SDF*SDFt

            x = cat((SDFt,a_1),-1)
            gates_PKA = self.linear_4(x)
            gates_PKA = self.sigmoid(gates_PKA)

            PKAt = self.linear_5(x)
            PKAt = self.tanh(PKAt)

            PKAt = gates_PKA*PKAt

            ins = cat((k,a_1,sd_1,qd_1),-1) 
            gates_KSU = self.linear_6(ins)
            gates_KSU = self.sigmoid(gates_KSU)

            k = gates_KSU*k + (1-gates_KSU)*PKAt

            h_i = unsqueeze(k,dim=1)
            h.append(h_i)

        output = cat(h,axis = 1)
        logits = sum(target_data*output,dim = -1)
        y = self.sigmoid(logits)
        
        return y
