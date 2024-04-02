from business_safety_classifier import BusinessSafetyClassifier
import os
from utils import get_args


def main():
    args = get_args()   

    assert args.lr_clf != None, 'No classifier path specified, please specify the path.'

    assert os.path.exists(args.lr_clf), 'Classifier path does not exist, please specify the correct path.'

    biz_safety_clf = BusinessSafetyClassifier(args)

    # run batch predictions
    biz_safety_clf.predict_batch(args)
    print('Batch predictions finished! Results saved to {}.'.format(args.filedir+args.output+'.csv'))


    # run single predictions
    testing_string = ["Every morning, I make a cup of coffee to start my day.",
                      "hello world",
                      'Jack likes to eat apples',
                       'revenue growed 20%']

    # for txt in testing_string:
    #     print('Text to be classified: ', txt)
    #     print('Predicted label: ', biz_safety_clf.predict(args, txt))




if __name__=="__main__":
    main()