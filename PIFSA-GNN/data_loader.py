import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class DataLoader:

    def __init__(self, data):
        self.cfg = {
            'movie20': {
                'item2id_path': 'data/movie20/item_index2entity_id.txt',
                'kg_path': 'data/movie20/kg.txt',
                'rating_path': 'data/movie20/ratings.txt',
                'rating_sep': '\t',
            },
            'music': {
                'item2id_path': 'data/music/item_index2entity_id.txt',
                'kg_path': 'data/music/kg.txt',
                'rating_path': 'data/music/ratings.txt',
                'rating_sep': '\t',
            },
            'restaurant': {
                'item2id_path': 'data/restaurant/item_index2entity_id.txt',
                'kg_path': 'data/restaurant/kg.txt',
                'rating_path': 'data/restaurant/ratings.txt',
                'rating_sep': '\t',
            },
        }
        self.data = data
        df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item','id'])
        df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head','relation','tail'])
        df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'], names=['userID', 'itemID', 'rating'], skiprows=1)
        self.df_item2id = df_item2id
        self.df_kg = df_kg
        self.df_rating = df_rating
        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()
        self._encoding()
        
    def _encoding(self):
        self.user_encoder.fit(self.df_rating['userID'])
        self.entity_encoder.fit(pd.concat([self.df_item2id['id'], self.df_kg['head'], self.df_kg['tail']]))
        self.relation_encoder.fit(self.df_kg['relation'])
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])

    def _build_dataset(self):
        df_dataset = pd.DataFrame()
        df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'])
        df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'])
        df_dataset['label'] = self.df_rating['rating']
        return df_dataset
        
    def _construct_kg(self):
        kg = dict()
        for i in tqdm(range(len(self.df_kg)), total=len(self.df_kg), desc='Create Kg'):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        return kg
        
    def load_dataset(self):
        return self._build_dataset()

    def load_kg(self):
        return self._construct_kg()
    
    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)
    
    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))
