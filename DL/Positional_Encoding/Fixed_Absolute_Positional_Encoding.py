import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class Divider: # 문장을 입력받아 단어 단위로 반환하는 클래스
    def __init__(self, sentence): # 문장 입력
        self.word = sentence.split(' ')
    def retIndex(self):
        return self.word # 단어 출력
    
class PE:
    def __init__(self, d_model, Pos): # 모델 차원, 단어 위치
        self.degree = np.linspace(0,360,361)
        self.d_model = d_model
        self.__Pos = Pos
    def get_value(self):
        dimension_list = []
        for i in range(self.d_model):
            if i %2 == 0:
                dimension_list.append(np.sin(self.__Pos/(10000**(2*i/self.d_model))))
            else:
                dimension_list.append(np.cos(self.__Pos/(10000**(2*i/self.d_model))))
        return dimension_list # 단어별 Positional Encoding 결과 반환
    
class Sketcher:
    def __init__(self, word):
        self.row = word.shape[0]
        self.col = word.shape[1]

    def showInfo(self):
        print(f"row: {self.row}, col: {self.col}")

    def plot(self):
        plt.figure(figsize=(32, 8))
        sns.heatmap(word, annot=False, cmap="coolwarm")
        plt.title("Positional Encoding")
        plt.show()

if __name__ == "__main__":

    sentence = """After heat records were smashed and a torrent of extreme weather events rocked countless countries in 2023, some climate scientists believed that the waning of the El Niño weather pattern could mean 2024 would be slightly cooler. It didn’t happen that way. This year is expected to break 2023’s global average temperature record and the effects of the warming – more powerful hurricanes, floods, wildfires and suffocating heat – have upended lives and livelihoods. All year, Associated Press photographers around the globe have captured moments, from the brutality unleashed during extreme weather events to human resilience in the face of hardship, that tell the story of a changing Earth.
January: Experiencing a changing world
As seas rise, salty ocean water of the Pacific encroaches on Vietnam’s Mekong Delta, hurting agriculture and the farmers and sellers who rely on it. Life for those on the Mekong now – paddling across markets and working and sleeping from houseboats – is quickly being altered. In Tahiti, the arrival of the Paris Olympics this year meant giant structures were built on one of their most precious reefs. The reefs sustain the life of sea creatures and in turn, the people of the island.
February: Farming against tougher odds
In many parts of the world, there were impacts when agriculture intersected with climate change. In Spain and other European countries, farmers were upset over increasing energy and fertilizer costs, cheaper farm imports entering the European Union and pesticide regulations, arguing all these changes could force them out of business. In Kenya, access to water continued to be a struggle for many, while fishers off the Indian coast of Mumbai had to contend with a rapidly warming Arabian Sea. There were bright spots, however, such as the increasing use of natural farming techniques that are more resistant to climate shocks.
March: Struggling to get water
More than 2 billion people around the world don’t have access to safely managed drinking water, according to the United Nations, a grim reality experienced in so many places. In Brazil, some residents collected water as it came down a mountain, while in India others filled up jugs from a street drain. Drinking from such sources can lead to many waterborne illnesses.
April: Fighting to thrive
For the Ojibwe tribe in the United States, spearfishing is an important tradition, one they maintained this year in the face of climate change. At the same time, in other parts of the world the impact of climate change was so severe that simply surviving was the best hope. Such was the case in Kenya, where floods took lives and forced many to evacuate, and in an Indian village where flooding is so constant that residents are constantly displaced.
May: Getting forced from home
When heavy rains led to massive flooding in Uruguay and Brazil, residents were forced from their homes. In both of these places, most people likely returned and were able to rebuild their lives. In other places, there was no going back. Such was the case for Quinault Indian Nation in the U.S., in the process of being relocated inland as coastal erosion threaten their homes. The Gardi Sugdub island off the coast of Panama faced a similar fate – hundreds of families are relocating to the mainland as sea levels rise.
June: Suffering from heat
From Mexico to Pakistan and beyond, high temperatures hit people hard. Unable to find relief, some sweated profusely while others ended up hospitalized. Many would die, such as in Saudi Arabia, where heat-related illnesses killed more than 1,300 during the annual hajj pilgrimage. The heat didn’t just impact people, but also oceans and animals, putting at risk some of the most biodiverse ecosystems in the world, such as Ecuador’s Galapagos Islands.
July: California burning
Rising temperatures and prolonged droughts create conditions for more and longer burning wildfires. One of the places that is consistently hard hit is the U.S. state of California. This year was no exception. Wildfires burned more than 1 million acres, chewed through hundreds of homes and led thousands of people to evacuate. As happens in every fire, countless animals also perished or were forced from their habitats.
August: Mother nature shining through
For all the destruction that climate change caused in 2024, mother nature showed off its beauty. That was on display at Churchill, Manitoba, a northern Canadian town that revels in its unofficial title as polar bear capital of the world. Like every year, tourists enjoyed stunning views of the Hudson Bay, watched beluga whales swim and, of course, came into contact with polar bears.
September: Raging waters
Water is central for humans and animals, but it can also take lives and leave a path of destruction. It did both in 2024. The scenes were shocking: students in India using rope to cross a flooded street, a little girl in Cuba floating in a container and Nigerians wading through floodwaters after a dam collapsed in the wake of heavy rains.
October: Experiencing extremes
Throughout the year, there was way too much water in some places and not enough in others, increasingly common as climate change alters natural weather patterns. In the Sahara Desert in Morocco, heavy rain left sand dunes with pools of water. By contrast, the Amazon region in South America, normally lush as a largely tropical area, experienced severe drought.
November: Astonishing destruction
Around the world, numerous storms unleashed powerful winds and dumped large amounts of water. The result: buildings and homes that looked like they had been hit with a wrecking ball, clothes and other household goods caked in mud and scattered on the ground, and residents walking through floodwaters.
December: Looking to 2025
As the end of 2024 approached, the arrival of winter in the Northern Hemisphere meant relief from the heat in the form of cold temperatures and idyllic scenes like snow‑frosted trees. But there were also reminders that global warming had already altered Earth so much that climate‑driven disasters, such as raging wildfires even during winter months, are never far off. While impossible to predict when and where disaster may strike, one thing is all but certain in 2025: the storms, floods, heat waves, droughts and wildfires will continue."""
    
    divided_sentence = Divider(sentence=sentence).retIndex()
    word = []

    for idx, each in enumerate(divided_sentence):
        word.append(PE(512, idx).get_value())
    word = np.array(word)
    # print((word.shape))
    
    sketch = Sketcher(word=word)
    sketch.showInfo()
    sketch.plot()









