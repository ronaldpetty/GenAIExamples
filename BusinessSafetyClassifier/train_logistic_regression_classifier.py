from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from utils import load_data_for_st_encoders, make_batches, get_args
from sentence_transformers import SentenceTransformer



def prepare_data(args):
    text, labels = load_data_for_st_encoders(args)
    # run btaches thru embedding model to get embeddings
    model = SentenceTransformer(model_name_or_path=args.model, trust_remote_code=True)
    n = 0
    for batch in make_batches(text, args.batch_size):
        # batch = add_prefix(batch, args.prefix)
        print(batch)
        embeddings = model.encode(batch, convert_to_tensor=True)
        
        embeddings = embeddings.cpu().detach().numpy()

        print('Shape of emb of this batch: ', embeddings.shape)

        if n == 0:
            all_embeddings = embeddings
        else:
            all_embeddings = np.append(all_embeddings, embeddings, axis = 0)
        
        n += 1

    print('Shape of all embedding: ', all_embeddings.shape)

    return all_embeddings, labels
    



def train_lr_classifier(args, data, labels):
    
    clf = LogisticRegression(random_state=args.random_seed,
                             max_iter=args.max_iter)
    clf.fit(data, labels)
    # accuracy on training set
    acc = clf.score(data, labels)
    print('Accuracy: {:.3f}'.format(acc))
    return clf

def save_classifier(clf, args):
    from joblib import dump
    dump(clf, args.output)
    print('Saved classifier at :', args.output)

def main():
    args = get_args()
    data, labels = prepare_data(args)
    clf = train_lr_classifier(args, data, labels)
    save_classifier(clf, args)

if __name__=="__main__":
    main()




