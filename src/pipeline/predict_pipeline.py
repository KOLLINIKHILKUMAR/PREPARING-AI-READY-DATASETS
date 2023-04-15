from my_imports import *
from src.components.data_transformation import know_type,Preprocessing,WordEmbedding,Classifier,heatmap
from src.exception import CustomException
import sys

class PredictPipeline:
    def __init__(self):
        pass

    def run(self,input_list):
        try:
            X = pd.read_csv(os.path.join("artifacts", "X-data.csv"),names=["text"], header=None)
            Y = pd.read_csv(os.path.join("artifacts", "Y-data.csv"),names=["label"], header=None)
            input_list[1].sort()
            preprocessor = Preprocessing(input_list[1],X)
            WordEmbedder = WordEmbedding(input_list[2],X,Y)
            classi = Classifier(input_list[3],WordEmbedder)
            heatmapoutput=heatmap(classi)
            image_exists, output = heatmapoutput.get_output()
            return image_exists,output
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        uploaded_file, 
        sep,
        preprocessing,
        word_embedding,
        classifier,
        ):
        self.uploaded_file = uploaded_file
        self.sep = sep
        self.preprocessing = preprocessing
        self.word_embedding = word_embedding
        self.classifier = classifier
        self.custom_data_input_list = []

    def get_data_as_list(self):
        try:
            self.custom_data_input_list = []
            self.custom_data_input_list.append(self.sep)
            self.custom_data_input_list.append(self.preprocessing)
            self.custom_data_input_list.append(self.word_embedding)
            self.custom_data_input_list.append(self.classifier)
            return self.custom_data_input_list
        except Exception as e:
            raise CustomException(e, sys)


    def get_df(self):
        self.uploaded_file.save(os.path.join("artifacts", "raw_data"))
        df_type=self.uploaded_file.content_type
        know=know_type()
        if df_type=='text/plain':
            know.if_text(self.custom_data_input_list[0])
        elif df_type=='application/pdf':
            know.if_pdf()
        elif df_type=='image/jpeg':
            know.if_image()
        else:
            raise CustomException("The file type is not supported",sys)






if __name__ == "__main__":
    pass