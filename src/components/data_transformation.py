from my_imports import *
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
class know_type:
    def __init__(self):
        pass
    def if_text(self,delimiter):
        df = pd.read_csv("artifacts/raw_data", sep=delimiter, header=None, names=["label", "text"], on_bad_lines='skip')
        df = df.fillna("")
        df[['text']].to_csv(os.path.join("artifacts", "X-data.csv"), header=None)
        df[['label']].to_csv(os.path.join("artifacts", "Y-data.csv"), header=None)
    def if_pdf():
        pass
    def if_image():
        pass

class Preprocessing:
    def __init__(self, names,X):
        self.names = names
        self.X=X
        self.call_name_functions()

    def call_name_functions(self):
        for name in self.names:
            if hasattr(self, name) and callable(getattr(self, name)):
                name_function = getattr(self, name)
                name_function()


    def stemming(self):
        stemmer=SnowballStemmer("english")
        print(self.X)
        self.X["text"] = self.X["text"].apply(lambda text: " ".join([stemmer.stem(word.lower()) for word in text.split()]))
        return self.X["text"]
    
    def lemmatization(self):
        wordnet_lemmatizer = WordNetLemmatizer()
        self.X["text"] = self.X["text"].apply(lambda text: " ".join([wordnet_lemmatizer.lemmatize(word) for word in text.split()]))
        return self.X
    
    def stopword_removal(self):
        nltk_stopwords = set(nltk.corpus.stopwords.words('english'))
        self.X["text"] = self.X["text"].apply(lambda text: " ".join([word.lower() for word in text.split() if not word in nltk_stopwords]))
        return self.X["text"]

    def punctuation_removal(self):
        punct_tag = re.compile(r'[^\w\s]')
        self.X["text"] = self.X["text"].apply(lambda text: punct_tag.sub(r'', text))
        return self.X

    def lower_case(self):
        self.X["text"] = self.X["text"].apply(lambda text: " ".join([word.lower() for word in text.split()]))
        return self.X

    def remove_numbers(self):
        tag = re.compile(r'[0-9]+')
        self.X["text"] = self.X["text"].apply(lambda text: tag.sub(r'', text))
        return self.X

    def remove_romans(self):
        html_tag_pattern = re.compile(r'<.*?>')
        en_tag = re.compile(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$')
        self.X["text"] = self.X["text"].apply(lambda text: html_tag_pattern.sub('', text))
        self.X["text"] =self.X["text"].apply(lambda text: en_tag.sub(r'', text))
        return self.X

    def redunable_words(self):
        red_tag = re.compile(r'[?<=(  )\\]|[&&|\|\|-]')
        self.X["text"] = self.X["text"].apply(lambda text: red_tag.sub(r' ', text))
        self.X["text"] = self.X["text"].str.replace(r'\s+', ' ')
        return self.X


class WordEmbedding:
    def __init__(self, name,X,Y):
        self.name = name
        self.X=X
        self.Y=Y
        self.call_name_function()

    def call_name_function(self):
        if hasattr(self, self.name) and callable(getattr(self, self.name)):
            name_function = getattr(self, self.name)
            name_function()

    def TFIDF(self):
        tfidf = TfidfVectorizer()
        self.X = tfidf.fit_transform(self.X["text"])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.3)
        self.X_train_vtc = self.X_train
        self.X_test_vtc = self.X_test
        createouputfile=createfile(self.X_train_vtc, self.X_test_vtc, self.Y)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def countvectorizer(self):
        pass

    def word2vec(self):
        pass

    def word2vec_pretrained(self):
        pass

    def glove(self):
        pass

    def fasttext(self):
        pass

class Classifier:
    def __init__(self, name,WordEmbedder):
        self.name = name
        self.WordEmbedder = WordEmbedder
        self.call_name_function()

    def call_name_function(self):
        if hasattr(self, self.name) and callable(getattr(self, self.name)):
            name_function = getattr(self, self.name)
            name_function()

    def random_forest(self):
        rf_rf = RandomForestClassifier()
        rf_model_rf = rf_rf.fit(self.WordEmbedder.X_train, self.WordEmbedder.y_train)
        y_pred = rf_model_rf.predict(self.WordEmbedder.X_test)
        self.score = classification_report(self.WordEmbedder.y_test, y_pred, output_dict=True)
        return self.score

    def decision_tree(self):
        pass

    def logistic_regression(self):
        pass

    def knn(self):
        pass

    def svm(self):
        pass

    def voting_classifier(self):
        pass

    def stacking_classifier(self):
        pass


class createfile:
    def __init__(self, X_train_vtc, X_test_vtc, Y):
        self.X_train_vtc = X_train_vtc
        self.X_test_vtc = X_test_vtc
        self.Y=Y
        self.filecreation(self.X_train_vtc, self.X_test_vtc, self.Y)

    def filecreation(self, X_train_vtc, X_test_vtc, Y):
        l = []
        l.extend(X_train_vtc)
        l.extend(X_test_vtc)
        my = pd.DataFrame(l)
        my["labels"] = Y
        my.to_csv(os.path.join("artifacts", "AI_ready_data.csv"), header=None,index=False)

class heatmap:
    def __init__(self, classi):
        self.classi=classi
        self.create_heatmap_plot()

    def create_heatmap_plot(self):
        plot = sns.heatmap(pd.DataFrame(self.classi.score).iloc[:-1, :].T, annot=True)
        plt.savefig(os.path.join("artifacts", "plot.png"))
        self.classi.score = pd.DataFrame(self.classi.score).transpose()
        self.classi.score.to_csv(os.path.join("artifacts", "classification_report.csv"))
        df = pd.read_csv(os.path.join("artifacts", "classification_report.csv"))
        self.output = df.to_dict(orient='records')
        self.image_path = os.path.join("artifacts", "plot.png")
        if os.path.exists(self.image_path):
            self.image_exists = True
        else:
            self.image_exists = False

    
    def get_output(self):
        return self.image_exists, self.output

if __name__ == "__main__":
    pass