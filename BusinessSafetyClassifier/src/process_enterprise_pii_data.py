import pandas as pd
import json
from utils import get_args


def process_text(text):
    temp = text.split('Passage: ')[-1]
    temp = temp.split('Choices')[0]
    processed = temp.strip('\n').strip('\'\'\'')
    if len(processed)>10:
        return processed
    else:
        return None


def main():
    args = get_args()
    print(args)

    text_list = []
    golden_labels = []
    text_length = []

    n = 0

    with open(args.filedir+args.filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            text=process_text(data['query'])
            if text:
                text_list.append(text)
                golden_labels.append(data['gold'])
                text_length.append(len(text))
                n += 1
                if len(text)<50:
                    print(text)
                    print('label: ', data['gold'])
                    print('unprocessed:\n', data['query'])
                    print('-'*50)
            


    print(len(text_list))
    print('max text length: {}, min text length: {}'.format(max(text_length), min(text_length)))

    df = pd.DataFrame({
        args.text_col: text_list,
        args.label_col: golden_labels,
        args.length_col: text_length,
    })

    df.to_csv(args.filedir+args.output+'.csv')
    df.to_json(args.filedir+args.output+'.jsonl',  orient='records', lines = True)

    print('unique labels: ', pd.unique(df['label']))

    print('Distribution of text length of entire dataset:')
    print(df.describe())
    


if __name__=="__main__":
    main()     