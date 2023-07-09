from predict_functions import *


def main():
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    prob, classes = predict(args.filepath, model, args.top_k, args.gpu)
    top_cat = [cat_to_name[x] for x in classes]
    
    print("************* Outputs *********************")
    print("The top five probabilities: {}".format(prob))
    print("The top five catagoies are: {}".format(top_cat))
    print("")
    print("The top choice is: {} with a probability of: {:.1f}%".format(top_cat[0], prob[0] * 100))


if __name__ == "__main__":
    main()