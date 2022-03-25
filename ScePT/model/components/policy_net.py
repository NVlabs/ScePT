import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.components import AdditiveAttention
from model.model_utils import *
import pdb
from collections import defaultdict



class clique_policy_net(nn.Module):
    def __init__(self,device,node_types,edge_types,input_dim, state_dim, z_dim, rel_state_fun, dyn_net, edge_encoding_net, 
                 edge_enc_dim , map_enc_dim, obs_lstm_hidden_dim,state_lstm_hidden_dim, FC_hidden_dim,input_scale, max_Nnode, dt, 
                 hyperparams, att_internal_dim = 16):
        super(clique_policy_net, self).__init__()
        self.device = device
        self.node_types = node_types
        self.edge_types = edge_types
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.dyn_net = dyn_net
        self.max_Nnode = max_Nnode
        self.obs_lstm_hidden_dim = obs_lstm_hidden_dim
        self.state_lstm_hidden_dim = state_lstm_hidden_dim
        self.edge_encoding_net = edge_encoding_net
        self.edge_enc_dim = edge_enc_dim
        self.rel_state_fun = rel_state_fun
        self.dt = dt
        if map_enc_dim is None:
            self.map_encoding = False
        else:
            self.map_encoding = True
            self.map_enc_dim  =map_enc_dim
        self.input_scale = input_scale
        self.obs_lstm_h0_net = nn.ModuleDict()
        self.obs_lstm_c0_net = nn.ModuleDict()
        self.state_lstm_h0_net = nn.ModuleDict()
        self.state_lstm_c0_net = nn.ModuleDict()
        self.obs_att = nn.ModuleDict()
        self.obs_lstm = nn.ModuleDict()
        self.state_lstm = nn.ModuleDict()
        self.action_net = nn.ModuleDict()
        for node_type in self.node_types:
            self.obs_att[node_type.name] = AdditiveAttention(encoder_hidden_state_dim=edge_enc_dim, decoder_hidden_state_dim=self.state_dim[node_type], internal_dim=att_internal_dim).to(self.device)
            self.obs_lstm[node_type.name] = nn.LSTM(edge_enc_dim,obs_lstm_hidden_dim).to(self.device)
            self.state_lstm[node_type.name] = nn.LSTM(self.state_dim[node_type]+self.input_dim[node_type],self.state_lstm_hidden_dim).to(self.device)
            self.obs_lstm_h0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.obs_lstm_hidden_dim)
            self.obs_lstm_c0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.obs_lstm_hidden_dim)
            self.state_lstm_h0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.state_lstm_hidden_dim)
            self.state_lstm_c0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.state_lstm_hidden_dim)
            if self.map_encoding and node_type in self.map_enc_dim:
                self.action_net[node_type.name] = simplelinear(obs_lstm_hidden_dim+state_lstm_hidden_dim+self.z_dim+self.map_enc_dim[node_type]+self.state_dim[node_type],input_dim[node_type],device,FC_hidden_dim,input_scale[node_type])
            else:
                self.action_net[node_type.name] = simplelinear(obs_lstm_hidden_dim+state_lstm_hidden_dim+self.z_dim+self.state_dim[node_type],input_dim[node_type],device,FC_hidden_dim,input_scale[node_type])
            
    def forward(self, batch_state_history,node_history_encoded,encoded_map, indices, ft, batch_z):
        node_index,edge_index,node_inverse_index,batch_node_to_edge_index,edge_to_node_index = indices

        batch_state_pred = dict()
        batch_input_pred = dict()
        batch_edge = dict()
        batch_obs = dict()
        obs_lstm_h = dict()
        obs_lstm_c = dict()
        state_lstm_h = dict()
        state_lstm_c = dict()
        batch_edge_enc = [None]*ft

        node_num = 0 
        for nt in self.node_types:
            node_num+=len(node_index[nt])


        for node_type in self.node_types:
            # batch_state_pred[node_type] = torch.zeros([ft,len(node_index[node_type]),self.state_dim[node_type]]).to(self.device)
            # batch_input_pred[node_type] = torch.zeros([ft,len(node_index[node_type]),self.input_dim[node_type]]).to(self.device)
            batch_state_pred[node_type] = [None]*ft
            batch_input_pred[node_type] = [None]*ft
        
        edge_to_obs_idx = dict()
        for et in self.edge_types:
            edge_to_obs_idx[et] = defaultdict(list)
            for idx, (node_idx,nb_idx) in edge_to_node_index[et].items():
                edge_to_obs_idx[et][node_idx].append(idx)


        # for edge_type in self.edge_types:
        #     batch_edge[edge_type] = torch.zeros([len(edge_index[edge_type]),self.state_dim[edge_type[0]]+self.state_dim[edge_type[1]]]).to(self.device)

        for nt in self.node_types:
            batch_state_pred[nt][0] = batch_state_history[nt][-1]
            obs_lstm_h[nt] = self.obs_lstm_h0_net[nt](batch_state_history[nt][-1]).view(1,len(node_index[nt]),self.obs_lstm_hidden_dim).to(self.device)
            obs_lstm_c[nt] = self.obs_lstm_c0_net[nt](batch_state_history[nt][-1]).view(1,len(node_index[nt]),self.obs_lstm_hidden_dim).to(self.device)
            state_lstm_h[nt] = self.state_lstm_h0_net[nt](batch_state_history[nt][-1]).view(1,len(node_index[nt]),self.obs_lstm_hidden_dim).to(self.device)
            state_lstm_c[nt] = self.state_lstm_c0_net[nt](batch_state_history[nt][-1]).view(1,len(node_index[nt]),self.obs_lstm_hidden_dim).to(self.device)


        batch_edge_idx1 = {et:list() for et in self.edge_types}
        batch_edge_idx2 = {et:list() for et in self.edge_types}
        for et in self.edge_types:
            for idx,(idxj,idxk) in batch_node_to_edge_index[et].items():
                batch_edge_idx1[et].append(idxj)
                batch_edge_idx2[et].append(idxk)

        
        node_obs_idx = {nt:torch.zeros([len(node_index[nt]),self.max_Nnode-1],dtype=torch.long) for nt in self.node_types}
        offset = 1
        edge_idx_offset = dict()

        for et in self.edge_types:
            edge_idx_offset[et] = offset
            offset+= len(edge_index[et])
        for et in self.edge_types:
            nt = et[0]
            for idx, (node_idx,nb_idx) in edge_to_node_index[et].items():
                node_obs_idx[nt][node_idx,nb_idx] = idx + edge_idx_offset[et]

        diff = 0
        for t in range(ft):
            batch_obs = dict()

            ## put states into raw obs tensor
            batch_edge_enc[t] = torch.zeros([1,self.edge_enc_dim]).to(self.device)
            for edge_type in self.edge_types:
                dim1 = self.state_dim[edge_type[0]]
                dim2 = self.state_dim[edge_type[1]]
                if t==0:
                    
                    batch_edge[edge_type] = torch.cat((batch_state_history[edge_type[0]][-1,batch_edge_idx1[edge_type]],\
                                                       batch_state_history[edge_type[1]][-1,batch_edge_idx2[edge_type]]),dim=1)
                    
                else:
                    batch_edge[edge_type] = torch.cat((batch_state_pred[edge_type[0]][t-1][batch_edge_idx1[edge_type]],\
                                                       batch_state_pred[edge_type[1]][t-1][batch_edge_idx2[edge_type]]),dim=1)
                    
            ## pass through pre-encoding network
                batch_edge_enc[t] = torch.cat((batch_edge_enc[t],self.edge_encoding_net[edge_type](batch_edge[edge_type][:,0:dim1],batch_edge[edge_type][:,dim1:dim1+dim2])),dim=0)
            ## put encoded vectors into observation matrices of each node
            for nt in self.node_types:
                batch_obs[nt] = batch_edge_enc[t][node_obs_idx[nt]]

            ## pass the observations through attention network and LSTM, and eventually action net
            for nt in self.node_types:
                if t==0:
                    rel_state = self.rel_state_fun[nt](batch_state_history[nt][-1],batch_state_history[nt][-1])
                    batch_u = torch.zeros([rel_state.shape[0],self.input_dim[nt]]).to(self.device)
                else:
                    rel_state = self.rel_state_fun[nt](batch_state_pred[nt][t-1],batch_state_history[nt][-1])
                    batch_u = batch_input_pred[nt][t-1]

                batch_obs_enc,_ = self.obs_att[nt](batch_obs[nt],rel_state)
                obs_lstm_out,(obs_lstm_h[nt],obs_lstm_c[nt]) = self.obs_lstm[nt](torch.unsqueeze(batch_obs_enc,dim=0),(obs_lstm_h[nt],obs_lstm_c[nt]))
                state_lstm_out,(state_lstm_h[nt],state_lstm_c[nt]) = self.state_lstm[nt](torch.unsqueeze(torch.cat((rel_state,batch_u),dim=-1),dim=0),(state_lstm_h[nt],state_lstm_c[nt]))
                if self.map_encoding and nt in self.map_enc_dim:
                    batch_input_pred[nt][t] = self.action_net[nt](torch.cat((rel_state,torch.squeeze(obs_lstm_out,dim=0),torch.squeeze(state_lstm_out,dim=0),encoded_map[nt],batch_z[nt]),dim=-1))
                else:
                    batch_input_pred[nt][t] = self.action_net[nt](torch.cat((rel_state,torch.squeeze(obs_lstm_out,dim=0),torch.squeeze(state_lstm_out,dim=0),batch_z[nt]),dim=-1))



                ## integrate forward the dynamics
                if t==0:
                    batch_state_pred[nt][t] = self.dyn_net[nt](batch_state_history[nt][-1],batch_input_pred[nt][t],self.dt)
                else:
                    batch_state_pred[nt][t] = self.dyn_net[nt](batch_state_pred[nt][t-1],batch_input_pred[nt][t],self.dt)

        for nt in self.node_types:
            batch_state_pred[nt] = torch.stack(batch_state_pred[nt])
            batch_input_pred[nt] = torch.stack(batch_input_pred[nt])

        return batch_state_pred,batch_input_pred



class clique_guided_policy_net(nn.Module):
    def __init__(self,device,node_types,edge_types,input_dim, state_dim, z_dim, rel_state_fun, collision_fun, dyn_net, edge_encoding_net, 
                 edge_enc_dim , map_enc_dim, history_enc_dim, obs_lstm_hidden_dim,guide_RNN_hidden_dim, FC_hidden_dim,input_scale, max_Nnode, dt, 
                 hyperparams, att_internal_dim = 16):
        super(clique_guided_policy_net, self).__init__()
        self.device = device
        self.node_types = node_types
        self.edge_types = edge_types
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.dyn_net = dyn_net
        self.max_Nnode = max_Nnode
        self.guide_RNN_hidden_dim = guide_RNN_hidden_dim
        self.obs_lstm_hidden_dim = obs_lstm_hidden_dim
        self.history_enc_dim = history_enc_dim
        self.edge_encoding_net = edge_encoding_net
        self.edge_enc_dim = edge_enc_dim
        self.rel_state_fun = rel_state_fun
        self.collision_fun = collision_fun
        self.dt = dt
        self.hyperparams = hyperparams
        if map_enc_dim is None:
            self.map_encoding = False
        else:
            self.map_encoding = True
            self.map_enc_dim  =map_enc_dim
        self.input_scale = input_scale

        self.obs_lstm_h0_net = nn.ModuleDict()
        self.obs_lstm_c0_net = nn.ModuleDict()
        self.guide_hidden_net = nn.ModuleDict()
        self.guide_RNN = nn.ModuleDict()
        self.RNN_proj_net = nn.ModuleDict()

        self.obs_att = nn.ModuleDict()
        self.obs_lstm = nn.ModuleDict()
        self.state_lstm = nn.ModuleDict()
        self.action_net = nn.ModuleDict()
        for node_type in self.node_types:
            self.obs_att[node_type.name] = AdditiveAttention(encoder_hidden_state_dim=edge_enc_dim, decoder_hidden_state_dim=self.state_dim[node_type], internal_dim=att_internal_dim).to(self.device)
            self.obs_lstm[node_type.name] = nn.LSTM(edge_enc_dim,obs_lstm_hidden_dim).to(self.device)
            
            self.obs_lstm_h0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.obs_lstm_hidden_dim)
            self.obs_lstm_c0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.obs_lstm_hidden_dim)
            self.action_net[node_type.name] = simplelinear(obs_lstm_hidden_dim+self.z_dim+self.state_dim[node_type]+4,input_dim[node_type],device,FC_hidden_dim,input_scale[node_type])

            self.RNN_proj_net[node_type.name] = nn.Linear(self.guide_RNN_hidden_dim,2)
            if self.map_encoding  and node_type in self.map_enc_dim:
                if self.hyperparams['use_lane_dec']:
                    self.guide_RNN[node_type.name] = nn.GRU(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type]+3,self.guide_RNN_hidden_dim).to(self.device)
                    self.guide_hidden_net[node_type.name] = nn.Linear(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type],self.guide_RNN_hidden_dim)
                else:
                    self.guide_RNN[node_type.name] = nn.GRU(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type],self.guide_RNN_hidden_dim).to(self.device)
                    self.guide_hidden_net[node_type.name] = nn.Linear(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type],self.guide_RNN_hidden_dim)
            else:
                self.guide_RNN[node_type.name] = nn.GRU(self.history_enc_dim+self.z_dim,self.guide_RNN_hidden_dim).to(self.device)
                self.guide_hidden_net[node_type.name] = nn.Linear(self.history_enc_dim+self.z_dim,self.guide_RNN_hidden_dim)
            
    def forward(self, batch_state_history,batch_state_history_st,node_history_encoded,encoded_map,batch_node_size,batch_lane_st, indices, ft, batch_z,robot_traj = None):
        node_index,edge_index,node_inverse_index,batch_node_to_edge_index,edge_to_node_index,batch_edge_idx1,batch_edge_idx2 = indices

        batch_state_pred = dict()
        batch_input_pred = dict()
        batch_state_pred_st = dict()
        batch_edge = dict()
        batch_obs = dict()
        obs_lstm_h = dict()
        obs_lstm_c = dict()
        batch_edge_enc = [None]*ft

        node_num = 0 
        for nt in self.node_types:
            node_num+=len(node_index[nt])
        tracking_error = 0
        collision_cost = 0
        for node_type in self.node_types:
            batch_state_pred[node_type] = [None]*ft
            batch_state_pred_st[node_type] = [None]*ft
            batch_input_pred[node_type] = [None]*ft
        
        edge_to_obs_idx = dict()
        for et in self.edge_types:
            edge_to_obs_idx[et] = defaultdict(list)
            for idx, (node_idx,nb_idx) in edge_to_node_index[et].items():
                edge_to_obs_idx[et][node_idx].append(idx)


        # for edge_type in self.edge_types:
        #     batch_edge[edge_type] = torch.zeros([len(edge_index[edge_type]),self.state_dim[edge_type[0]]+self.state_dim[edge_type[1]]]).to(self.device)

        des_traj = dict()
        for nt in self.node_types:

            obs_lstm_h[nt] = self.obs_lstm_h0_net[nt](batch_state_history_st[nt][-1]).view(1,len(node_index[nt]),self.obs_lstm_hidden_dim).to(self.device)
            obs_lstm_c[nt] = self.obs_lstm_c0_net[nt](batch_state_history_st[nt][-1]).view(1,len(node_index[nt]),self.obs_lstm_hidden_dim).to(self.device)
            

            if self.map_encoding and nt in self.map_enc_dim:
                xz = torch.cat((node_history_encoded[nt][-1],encoded_map[nt],batch_z[nt]),dim=-1)
            else:
                xz = torch.cat((node_history_encoded[nt][-1],batch_z[nt]),dim=-1)
            
            if self.hyperparams['use_lane_dec'] and self.map_encoding and nt in self.map_enc_dim:
                # guide_RNN_hidden = self.guide_hidden_net[nt](torch.unsqueeze(xz,0))
                # des_traj[nt] = list()
                # for t in range(ft):

                #     if t==0:
                #         wp = torch.zeros([xz.shape[0],3]).to(self.device)
                #     elif t==1:
                        
                #         psi = torch.atan2(des_traj[nt][0][:,1],des_traj[nt][0][:,0]).reshape(-1,1)
                #         wp = torch.cat((des_traj[nt][0],psi),dim=-1)
                #     else:
                #         psi = torch.atan2(des_traj[nt][t-1][:,1]-des_traj[nt][t-2][:,1],des_traj[nt][t-1][:,0]-des_traj[nt][t-2][:,0]).reshape(-1,1)
                #         wp = torch.cat((des_traj[nt][t-1],psi),dim=-1)
                #     delta_y,delta_psi,ref_pt = batch_proj(wp,batch_lane_st[nt][...,[0,1,3]].permute(1,0,2))
                #     if delta_y.isnan().any() or delta_psi.isnan().any() or ref_pt.isnan().any():
                #         pdb.set_trace()
                #     ref_psi = ref_pt[...,2:3]
                #     RNN_input = torch.unsqueeze(torch.cat((xz,delta_y,torch.cos(ref_psi),torch.sin(ref_psi)),dim=-1),dim=0)
                #     RNN_out,guide_RNN_hidden = self.guide_RNN[nt](RNN_input,guide_RNN_hidden)
                #     des_traj[nt].append((self.RNN_proj_net[nt](RNN_out[0])))
                
                # des_traj[nt] = torch.stack(des_traj[nt],dim=0)
                xz = torch.unsqueeze(xz,0)
                xz_seq = torch.cat((xz.repeat(ft,1,1),batch_lane_st[nt][...,[0,1,3]]),dim=-1)
                guide_RNN_hidden = self.guide_hidden_net[nt](xz)
                RNN_out,_ = self.guide_RNN[nt](xz_seq,guide_RNN_hidden)
                des_traj[nt] = self.RNN_proj_net[nt](RNN_out)

            else:
                xz = torch.unsqueeze(xz,0)
                xz_seq = xz.repeat(ft,1,1)
                guide_RNN_hidden = self.guide_hidden_net[nt](xz)
                RNN_out,_ = self.guide_RNN[nt](xz_seq,guide_RNN_hidden)
                des_traj[nt] = self.RNN_proj_net[nt](RNN_out)

        
        node_obs_idx = {nt:torch.zeros([len(node_index[nt]),self.max_Nnode-1],dtype=torch.long) for nt in self.node_types}
        offset = 1
        edge_idx_offset = dict()

        for et in self.edge_types:
            edge_idx_offset[et] = offset
            offset+= len(edge_index[et])
        for et in self.edge_types:
            nt = et[0]
            for idx, (node_idx,nb_idx) in edge_to_node_index[et].items():
                node_obs_idx[nt][node_idx,nb_idx] = idx + edge_idx_offset[et]


        for t in range(ft):
            batch_obs = dict()

            ## put states into raw obs tensor
            batch_edge_enc[t] = torch.zeros([1,self.edge_enc_dim]).to(self.device)
            for edge_type in self.edge_types:
                dim1 = self.state_dim[edge_type[0]]
                dim2 = self.state_dim[edge_type[1]]
                try:
                    edge_node_size1 = batch_node_size[edge_type[0]][batch_edge_idx1[edge_type]]
                    edge_node_size2 = batch_node_size[edge_type[1]][batch_edge_idx2[edge_type]]
                except:

                    edge_node_size1 = batch_node_size[edge_type[0]][batch_edge_idx1[edge_type]]
                    edge_node_size2 = batch_node_size[edge_type[1]][batch_edge_idx2[edge_type]]

                if t==0:
                    batch_edge[edge_type] = torch.cat((batch_state_history[edge_type[0]][-1,batch_edge_idx1[edge_type]],\
                                                       batch_state_history[edge_type[1]][-1,batch_edge_idx2[edge_type]]),dim=1) 
                else:

                    batch_edge[edge_type] = torch.cat((batch_state_pred[edge_type[0]][t-1][batch_edge_idx1[edge_type]],\
                                                       batch_state_pred[edge_type[1]][t-1][batch_edge_idx2[edge_type]]),dim=1)

                if batch_edge[edge_type].shape[0]>0:
                    collision_cost += torch.sum(self.collision_fun[edge_type](batch_edge[edge_type][:,0:dim1],batch_edge[edge_type][:,dim1:dim1+dim2],edge_node_size1,edge_node_size2))
            ## pass through pre-encoding network
                if batch_edge[edge_type].shape[0]>0:
                    batch_edge_enc[t] = torch.cat((batch_edge_enc[t],self.edge_encoding_net[edge_type](batch_edge[edge_type][:,0:dim1],batch_edge[edge_type][:,dim1:dim1+dim2],edge_node_size1,edge_node_size2)),dim=0)
                    if batch_edge_enc[t].isnan().any():
                        pdb.set_trace()
            ## put encoded vectors into observation matrices of each node
            for nt in self.node_types:
                if node_obs_idx[nt].shape[0]>0:
                    # if (node_obs_idx[nt]==0).all():
                    #     pdb.set_trace()
                    try:
                        batch_obs[nt] = batch_edge_enc[t][node_obs_idx[nt]]
                    except:
                        batch_obs[nt] = batch_edge_enc[t][node_obs_idx[nt]]


            ## pass the observations through attention network and LSTM, and eventually action net
            for nt in self.node_types:
                if len(node_index[nt])>0:
                    if t==0:
                        rel_state = batch_state_history_st[nt][-1]
                        next_wp = des_traj[nt][t]-batch_state_history_st[nt][-1][:,0:2]
                    else:
                        rel_state = self.rel_state_fun[nt](batch_state_pred_st[nt][t-1],batch_state_history_st[nt][-1])
                        rel_state[...,0:2]-=des_traj[nt][t-1]
                        next_wp = des_traj[nt][t]-batch_state_pred_st[nt][t-1][:,0:2]

                    tracking_error += torch.linalg.norm(rel_state[...,0:2])/rel_state.shape[0]

                    batch_obs_enc,_ = self.obs_att[nt](batch_obs[nt],rel_state)
                    if batch_obs_enc.isnan().any():
                        pdb.set_trace()
                    obs_lstm_out,(obs_lstm_h[nt],obs_lstm_c[nt]) = self.obs_lstm[nt](torch.unsqueeze(batch_obs_enc,dim=0),(obs_lstm_h[nt],obs_lstm_c[nt]))
                    
                    if obs_lstm_out.isnan().any():
                        pdb.set_trace()
                    batch_input_pred[nt][t] = self.action_net[nt](torch.cat((rel_state,next_wp,batch_node_size[nt],torch.squeeze(obs_lstm_out,dim=0),batch_z[nt]),dim=-1))
                    if not robot_traj is None:
                        for idx,_ in robot_traj[nt].items():
                            batch_input_pred[nt][t][idx]=torch.zeros(self.input_dim[nt]).to(self.device)

                    ## integrate forward the dynamics
                    if t==0:
                        batch_state_pred[nt][t] = self.dyn_net[nt](batch_state_history[nt][-1],batch_input_pred[nt][t],self.dt)
                        batch_state_pred_st[nt][t] = self.dyn_net[nt](batch_state_history_st[nt][-1],batch_input_pred[nt][t],self.dt)
                    else:
                        batch_state_pred[nt][t] = self.dyn_net[nt](batch_state_pred[nt][t-1],batch_input_pred[nt][t],self.dt)
                        batch_state_pred_st[nt][t] = self.dyn_net[nt](batch_state_pred_st[nt][t-1],batch_input_pred[nt][t],self.dt)

                    if batch_state_pred[nt][t].isnan().any() or batch_state_pred_st[nt][t].isnan().any():
                        pdb.set_trace()

                    if not robot_traj is None:
                        for idx,(traj,traj_st) in robot_traj[nt].items():
                            if not (traj[t]==0).all():
                                batch_state_pred[nt][t][idx] = traj[t]
                                batch_state_pred_st[nt][t][idx] = traj_st[t]
                else:
                    batch_state_pred[nt][t] = torch.zeros([0,self.state_dim[nt]]).to(self.device)
                    batch_state_pred_st[nt][t] = torch.zeros([0,self.state_dim[nt]]).to(self.device)
                    batch_input_pred[nt][t] = torch.zeros([0,self.input_dim[nt]]).to(self.device)
            

        for nt in self.node_types:

            batch_state_pred[nt] = torch.stack(batch_state_pred[nt])
            batch_state_pred_st[nt] = torch.stack(batch_state_pred_st[nt])
            batch_input_pred[nt] = torch.stack(batch_input_pred[nt])

        return batch_state_pred,batch_state_pred_st,batch_input_pred,des_traj,tracking_error,collision_cost/node_num

    def forward_clique(self, clique_type,clique_state_history,clique_state_history_st,clique_history_encoded,encoded_map,clique_node_size, ft, clique_z,robot_traj = None):
        N = len(clique_type)
        node_state_pred = [None]*N 
        node_input_pred = [None]*N 
        edge = dict()
        lstm_h = [None]*N 
        lstm_c = [None]*N 
        obs = [None]*N
        rel_state = [None]*N
        next_wp = [None]*N
        des_traj = [None]*N

        for i in range(N):
            node_state_pred[i]=[None]*ft
            node_input_pred[i]=[None]*ft
            lstm_h[i] = self.obs_lstm_h0_net[clique_type[i].name](clique_state_history_st[i][-1]).view(1,1,self.obs_lstm_hidden_dim).to(self.device)
            lstm_c[i] = self.obs_lstm_c0_net[clique_type[i].name](clique_state_history_st[i][-1]).view(1,1,self.obs_lstm_hidden_dim).to(self.device)
            if self.map_encoding and clique_type[i] in self.map_enc_dim:
                xz = torch.cat((clique_history_encoded[i],encoded_map[i],torch.unsqueeze(clique_z[i],dim=0)),dim=-1)
            else:
                xz = torch.cat((clique_history_encoded[i],torch.unsqueeze(clique_z[i],dim=0)),dim=-1)
            xz = torch.unsqueeze(xz,0)
            xz_seq = xz.repeat(ft,1,1)
            guide_RNN_hidden = self.guide_hidden_net[clique_type[i]](xz)
            RNN_out,_ = self.guide_RNN[clique_type[i]](xz_seq,guide_RNN_hidden)
            des_traj[i] = self.RNN_proj_net[clique_type[i]](RNN_out)

        for t in range(ft):
            if t==0:
                x = [clique_state_history[i][-1] for i in range(N)]
            else:
                x = [node_state_pred[i][t-1] for i in range(N)]
            for i in range(0,N):
                if t==0:
                    rel_state[i] = clique_state_history_st[i][-1]
                    next_wp[i] = des_traj[i][t,0]-clique_state_history_st[i][-1,0:2]
                else:
                    rel_state[i] = self.rel_state_fun[clique_type[i]](node_state_pred[i][t-1],clique_state_history_st[i][-1])
                    rel_state[i][0:2]-=des_traj[i][t-1,0]
                    next_wp[i] = des_traj[i][t,0]-node_state_pred[i][t-1][0:2]
                for j in range(0,N):
                    if i!=j:
                        edge[(i,j)] = self.edge_encoding_net[(clique_type[i],clique_type[j])](x[i],x[j],clique_node_size[i],clique_node_size[j])

                obs_ensemble = [edge[(i,j)] for j in range(0,N) if j!=i]
                if len(obs_ensemble)>0:
                    obs[i],_ = self.obs_att[clique_type[i].name](torch.unsqueeze(torch.stack(obs_ensemble,dim=0),dim=0),rel_state[i].view(1,-1))
                else:
                    obs[i] = torch.zeros([1,self.edge_enc_dim]).to(self.device)
                obs_lstm_out,(lstm_h[i],lstm_c[i]) = self.obs_lstm[clique_type[i].name](torch.unsqueeze(obs[i],dim=0),(lstm_h[i],lstm_c[i]))

                node_input_pred[i][t] = self.action_net[clique_type[i]](torch.cat((rel_state[i],next_wp[i],clique_node_size[i],torch.squeeze(obs_lstm_out),clique_z[i]),dim=-1))
                if t==0:
                    node_state_pred[i][t] = self.dyn_net[clique_type[i]](clique_state_history[i][-1],node_input_pred[i][t],self.dt)
                else:
                    node_state_pred[i][t] = self.dyn_net[clique_type[i]](node_state_pred[i][t-1],node_input_pred[i][t],self.dt)


        return node_state_pred,node_input_pred

