import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms
import numpy as np



class metadataset(Dataset):
    '''
    * train.npz
    * test.npz
    * val.npz
    '''

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        '''
        :param root: root dir path
        :param mode: train/test/val
        :param batchsz: batch size of sets
        :param n_way: n class
        :param k_shot: k data per each class
        :param k_query: target(personalized) number of samples per set for evaluation
        :param resize: resize
        :param startidx: start indx
        '''
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.setsz = self.n_way*self.k_shot
        self.querysz = self.n_way*self.k_query
        self.resize = resize
        self.startidx = startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
            mode, batchsz, n_way, k_shot, k_query, resize))

        if mode is 'train':
            self.dataset = np.load("./COHFACE_train_set_1~28_.npz")
            print("complete load train set")
        elif mode is 'val':
            self.dataset = np.load("./COHFACE_val_set_29~36_.npz")
            print("complete load val set")
        elif mode is 'test':
            self.dataset = np.load("./COHFACE_test_set_37~40_.npz")
            print("complete load test set")
        # dataset = bvpdataset(A=dataset['A'], M=dataset['M'], T=dataset['T'],N=dataset['N'])
        # A : Appearance M : Motion T : Target N : Person NUmber
        number = self.dataset['N']
        self.cls = list(set(self.dataset['N']))
        self.cls_num = len(self.cls)
        self.data_cnt = [0,]
        for num in self.cls:
            self.data_cnt.append(len(np.where(number == num)[0]))
        self.data_cnt = np.nancumsum(self.data_cnt)
        self.create_batch(self.batchsz)

    def create_batch(self, batchsz):
        '''
        :param batchsz: batch size
        :return:
        '''

        self.support_A_batch = [] # support Appearance dataset # Dtrain
        self.support_M_batch = [] # support Motion dataset
        self.suuport_T_batch = [] # support Target dataset
        self.query_A_batch = [] # query Appearance dataset # Dtest
        self.query_M_batch = [] # query Motion dataset
        self.query_T_batch = [] # query Target dataset

        for b in range(batchsz):
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)
            np.random_shuffle(selected_cls)
            support_A = []
            support_M = []
            support_T = []
            query_A = []
            query_M = []
            query_T = []
            for cls in selected_cls: # Support 80% query 20%
                data_len = self.data_cnt[cls] - self.data_cnt[cls-1]
                support_A.append(self.dataset['A'][self.data_cnt[cls-1]:self.data_cnt[cls-1]+data_len//10*8])
                support_M.append(self.dataset['M'][self.data_cnt[cls - 1]:self.data_cnt[cls - 1] + data_len // 10 * 8])
                support_T.append(self.dataset['T'][self.data_cnt[cls - 1]:self.data_cnt[cls - 1] + data_len // 10 * 8])
                query_A.append(self.dataset['A'][self.data_cnt[cls-1]+data_len//10*8:self.data_cnt[cls]])
                query_M.append(self.dataset['M'][self.data_cnt[cls - 1] + data_len // 10 * 8:self.data_cnt[cls]])
                query_T.append(self.dataset['T'][self.data_cnt[cls - 1] + data_len // 10 * 8:self.data_cnt[cls]])
            self.support_A_batch.append(support_A)
            self.support_M_batch.append(support_M)
            self.support_T_batch.append(support_T)
            self.query_A_batch.append(query_A)
            self.query_M_batch.append(query_M)
            self.query_T_batch.append(query_T)
        #   for cls in selected_cls:# k-shot``

        '''
        shuffle x
        '''
        # np.random.shuffle(support_A)
        # np.random.shuffle(support_M)
        # np.random.shuffle(query_A)
        # np.random.shuffle(query_M)

    def __getitem__(self, N):
        '''

        :param N: data number( Target person number )
        :return: Nth Person's data( Appearance, Motion, Target x Query & Support )
        '''
        data_len = self.data_cnt[N] - self.data_cnt[N - 1]
        support_A = torch.FloatTensor(self.dataset['A'][self.data_cnt[N-1]:self.data_cnt[N-1]+data_len//10*8])
        support_M = torch.FloatTensor(self.dataset['M'][self.data_cnt[N-1]:self.data_cnt[N-1]+data_len//10*8])
        support_t = torch.FloatTensor(self.dataset['T'][self.data_cnt[N-1]:self.data_cnt[N-1]+data_len//10*8])
        query_A = torch.FloatTensor(self.dataset['A'][self.data_cnt[N-1]+data_len//10*8:self.data_cnt[N]])
        query_M = torch.FloatTensor(self.dataset['M'][self.data_cnt[N-1]+data_len//10*8:self.data_cnt[N]])
        query_t = torch.FloatTensor(self.dataset['T'][self.data_cnt[N-1]+data_len//10*8:self.data_cnt[N]])

        return support_A,support_M,support_t,query_A,query_M,query_t



