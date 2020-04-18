# Integration test for FeatureGen class
from FeatureGen import FeatureGen

document = [['@', 'भारत', 'में', 'राष्ट्रवाद।'],
['जैसा', 'कि', 'आप', 'देख', 'चुके', 'है,', 'यूरोप', 'में', 'आधुनिक', 'राष्ट्रवाद', 'के', 'साथ', 'ही', 'राष्ट्र-राज्यों', 'का', 'भी', 'उदय', 'हुआ'],
['भारत', 'में','इससे', 'अपने', 'बारे', 'में', 'लोगों', 'की', 'समझ', 'बदलने', 'लगी'],
['वे', 'कौन', 'हैं,', 'उनकी', 'पहचान', 'किस', 'बात', 'से', 'परिभाषित', 'होती', 'है,', 'यह', 'भावना', 'बदल', 'गई'],
['उनमें', 'राष्ट्र', 'के', 'प्रति', 'लगाव', 'का', 'भाव', 'पैदा', 'होने', 'लगा'],
['नए', 'प्रतीकों', 'और', 'चिन्हों', 'ने,', 'नए', 'गीतों', 'और', 'विचारों', 'ने', 'नए', 'संपर्क', 'स्थापित', 'किए', 'और', 'समुदायों', 'की', 'सीमाओं', 'को', 'दोबारा', 'परिभाषित', 'कर', 'दिया'],
['ज़्यादातर', 'देशों', 'में', 'इस', 'नई', 'राष्ट्रपैय', 'पहचान', 'का', 'निर्माण', 'एक', 'लंबी', 'प्रक्रिया', 'में', 'हुआ'],
['आइए', 'देखें', 'कि', 'हमारे', 'देश', 'में', 'यह', 'चेतना', 'किस', 'तरह', 'पैदा', 'हुई?']]

document2 = [['।@', 'भारत', 'में', 'राष्ट्रवाद।'],
['भारत', 'में','इससे', 'अपने', 'बारे', 'में', 'लोगों', 'की', 'समझ', 'बदलने', 'लगी']]

test_object = FeatureGen(document)
feature_vector = test_object.getfeaturevec(document)



print(feature_vector)


