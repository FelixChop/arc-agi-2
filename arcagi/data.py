import os
import json

class Arcagi2:

    def get_data(self, add_solutions:bool=True) -> dict:
        data = self.get_raw_data()
        if not add_solutions:
            return data
        for s in ['training', 'evaluation']:
            s_chal = s+'_challenges'
            s_sol = s+'_solutions'
            for key in data[s_chal].keys():
                for i in range(len(data[s_chal][key]['test'])):
                    data[s_chal][key]['test'][i]['output'] = data[s_sol][key][i]
            del data[s_sol]
        return data

    def get_raw_data(self):
        data_path = os.path.join(os.path.dirname(__file__), '../data')
        file_names = [f for f in os.listdir(data_path) if f.startswith('arc-agi_')]
        key_names = [f_name.replace('.json','').replace('arc-agi_','') for f_name in file_names]
        return {
            key:self.load_json(os.path.join(data_path,f_name))
            for key,f_name in zip(key_names, file_names)
        }

    def load_json(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        return data

if __name__=='__main__':
    data = Arcagi2().get_data()
    print({key:len(data[key]) for key in data.keys()})
