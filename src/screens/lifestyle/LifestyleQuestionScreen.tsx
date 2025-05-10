import React, { useState, useEffect } from 'react';
import { View, SafeAreaView, TouchableOpacity, StyleSheet } from 'react-native';
import Modal from 'react-native-modal';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { saveLifestyle } from '../../api/lifestyleApi.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const LifestyleQuestionScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'LifestyleQuestion'>;
  const navigation = useNavigation<Navigation>();

  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);

  const [backConfirmModal, setbackConfirmModal] = useState(false);
  const [saveFailModal, setSaveFailModal] = useState(false);
  const [authFailModal, setAuthFailModal] = useState(false);

  const lifestyleQuestion = [
    {
      question: '가구원이 몇 명인가요?',
      options: ['1인', '2인', '3인', '4인', '5인 이상'],
    },
    {
      question: '배우자가 있나요?',
      options: ['있음', '없음'],
    },
    {
      question: '월 소득이 얼마인가요?',
      options: ['100만원 이하', '200만원 이하', '300만원 이하', '300만원 초과'],
    },
    {
      question: '월 지출이 얼마인가요?',
      options: ['100만원 이하', '200만원 이하', '300만원 이하', '300만원 초과'],
    },
    {
      question: '한 달 동안 음주에 지출한 비용이 얼마인가요?',
      options: ['5만원 이하', '10만원 이하', '15만원 이하', '20만원 초과'],
    },
    {
      question: '한 달 동안 담배에 지출한 비용이 얼마인가요?',
      options: ['5만원 이하', '10만원 이하', '15만원 이하', '20만원 초과'],
    },
    {
      question: '한 달 동안 서적에 지출한 비용이 얼마인가요?',
      options: ['5만원 이하', '10만원 이하', '15만원 이하', '20만원 초과'],
    },
    {
      question: '한 달 동안 의료에 지출한 비용이 얼마인가요?',
      options: ['5만원 이하', '10만원 이하', '15만원 이하', '20만원 초과'],
    },
    {
      question: '한 달 동안 보험에 지출한 비용이 얼마인가요?',
      options: ['5만원 이하', '10만원 이하', '15만원 이하', '20만원 초과'],
    },
    {
      question: '애완동물이 있나요?',
      options: ['있음', '없음'],
    },
    {
      question: '현재 가정의 형태가 어떻게 되나요?',
      options: ['일반 가정', '결손 가정', '나홀로 가정', '저소득 가정', '저소득 결손 가정', '저소득 나홀로 가정'],
    },
  ];

  const currentQuestion = lifestyleQuestion[currentQuestionIndex];
  const selectedOption = selectedOptions[currentQuestionIndex] || null;

  useEffect(() => {
    if (authFailModal) {
      const timer = setTimeout(() => {
        navigation.navigate('Login');
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [navigation, authFailModal]);

  const handleSubmit = async () => {
    try {
      const questionList = lifestyleQuestion.map((value, index) => ({
        question_id: index + 1,
        answer: selectedOptions[index],
      }));

      const response = await saveLifestyle(questionList);

      if (response.message === '라이프스타일 저장 성공') {
        navigation.navigate('LifestyleComplete');
      } else if (response.message === '라이프스타일 저장 실패') {
        console.log('라이프스타일 저장 실패');
        setSaveFailModal(true);
      } else {
        console.log('인증 실패');
        setAuthFailModal(true);
      }
    } catch (error) {
      console.log('라이프스타일 저장 실패');
      setSaveFailModal(true);
    }
  };

  return (
    <View style={styles.container}>
      <SafeAreaView>
        <View style={styles.headerContainer}>
          <TouchableOpacity onPress={() => {
            if (currentQuestionIndex !== 0) {
              setCurrentQuestionIndex((prev) => prev - 1);
            } else {
              setbackConfirmModal(true);
            }
          }}>
            <Icon name="chevron-back" size={16} />
          </TouchableOpacity>
        </View>
      </SafeAreaView>

      <View style={styles.contentContainer}>
        <CustomText style={styles.questionText}>{currentQuestion.question}</CustomText>
        {currentQuestion.options.map((option, index) => (
          <TouchableOpacity
            key={index}
            style={styles.optionContainer}
            onPress={() => {
              const updated = [...selectedOptions];
              updated[currentQuestionIndex] = option;
              setSelectedOptions(updated);
            }}
          >
            <CustomText style={styles.optionText}>{option}</CustomText>
            <Icon name="checkmark-circle-outline" size={24} color={selectedOption === option ? '#9F8473' : '#B4B4B4'} />
          </TouchableOpacity>
        ))}
      </View>

      <View style={styles.buttonWrapper}>
        <TouchableOpacity
          disabled={!selectedOption}
          style={[styles.nextButton, selectedOption ? styles.selectedButton : styles.unselectedButton]}
          onPress={() => {
            if (currentQuestionIndex < lifestyleQuestion.length - 1) {
              setCurrentQuestionIndex((prev) => prev + 1);
            } else {
              handleSubmit();
            }
          }}
        >
          <CustomText style={selectedOption ? styles.selectedText : styles.unselectedText}>
            {currentQuestionIndex !== lifestyleQuestion.length - 1 ? '다음' : '완료'}
          </CustomText>
        </TouchableOpacity>
      </View>

      {/* <Modal
        transparent={true}
        animationType="fade"
        visible={backConfirmModal}
        onRequestClose={() => setbackConfirmModal(false)}
      >
        <TouchableWithoutFeedback onPress={() => setbackConfirmModal(false)}>
          <View style={styles.modalOverlay}>
            <TouchableWithoutFeedback>
              <View style={styles.modalContent}>
                <CustomText style={styles.modalTitle}>라이프스타일 입력을 그만할까요?</CustomText>

                <View style={styles.modalButtonWrapper}>
                  <TouchableOpacity style={[styles.modalButton, styles.modalButtonPrimary]} onPress={() => navigation.navigate('Home')}>
                    <CustomText style={styles.modalButtonText}>네, 그만할래요</CustomText>
                  </TouchableOpacity>

                  <TouchableOpacity style={[styles.modalButton, styles.modalButtonSecondary]} onPress={() => setbackConfirmModal(false)}>
                    <CustomText style={styles.modalButtonText}>아니요, 계속 할래요</CustomText>
                  </TouchableOpacity>
                </View>
              </View>
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
      </Modal> */}

      <Modal
        isVisible={backConfirmModal}
        onBackdropPress={() => setbackConfirmModal(false)}
        onBackButtonPress={() => setbackConfirmModal(false)}
        backdropColor="rgb(69, 69, 69)"
        backdropOpacity={0.3}
        animationIn="fadeIn"
        animationOut="fadeOut"
        style={styles.modalOverlay}
      >
        <View style={styles.modalContent}>
          <CustomText style={styles.modalTitle}>라이프스타일 입력을 그만할까요?</CustomText>

          <View style={styles.modalButtonWrapper}>
            <TouchableOpacity style={[styles.modalButton, styles.modalButtonPrimary]} onPress={() => navigation.navigate('Home')}>
              <CustomText style={styles.modalButtonText}>네, 그만할래요</CustomText>
            </TouchableOpacity>

            <TouchableOpacity style={[styles.modalButton, styles.modalButtonSecondary]} onPress={() => setbackConfirmModal(false)}>
              <CustomText style={styles.modalButtonText}>아니요, 계속 할래요</CustomText>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <Modal
        isVisible={saveFailModal}
        onBackdropPress={() => setSaveFailModal(false)}
        onBackButtonPress={() => setSaveFailModal(false)}
        backdropColor="rgb(69, 69, 69)"
        backdropOpacity={0.3}
        animationIn="fadeIn"
        animationOut="fadeOut"
        style={styles.modalOverlay}
      >
        <View style={styles.modalContent}>
          <CustomText style={styles.modalTitle}>라이프스타일 저장에 실패했어요.</CustomText>

          <View style={styles.modalButtonWrapper}>
            <TouchableOpacity
              style={[styles.modalButton, styles.modalButtonPrimary]}
              onPress={() => {
                setSaveFailModal(false);
                handleSubmit();
              }}
            >
              <CustomText style={styles.modalButtonText}>다시 시도</CustomText>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.modalButton, styles.modalButtonSecondary]}
              onPress={() => {
                setSaveFailModal(false);
              }}
            >
              <CustomText style={styles.modalButtonText}>닫기</CustomText>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <Modal
        isVisible={authFailModal}
        onBackdropPress={() => setAuthFailModal(false)}
        onBackButtonPress={() => setAuthFailModal(false)}
        backdropColor="rgb(69, 69, 69)"
        backdropOpacity={0.3}
        animationIn="fadeIn"
        animationOut="fadeOut"
        style={styles.modalOverlay}
      >
        <View style={styles.modalContent}>
          <CustomText style={styles.modalContentText}>인증에 실패했어요.</CustomText>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    backgroundColor: '#FFFFFF',
  },
  headerContainer: {
    width: 70,
    paddingVertical: 16,
    paddingHorizontal: 10,
  },
  contentContainer: {
    paddingTop: 50,
    paddingHorizontal: 15,
  },
  questionText: {
    minHeight: 60,
    marginBottom: 50,
    fontSize: 24,
  },
  optionContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
  },
  optionText: {
    paddingBottom: 10,
    fontSize: 20,
    color: '#434240',
  },
  buttonWrapper: {
    alignItems: 'center',
    paddingBottom: 16,
    paddingHorizontal: 10,
    marginTop: 'auto',
  },
  nextButton: {
    alignItems: 'center',
    width: '100%',
    paddingVertical: 10,
    borderRadius: 10,
  },
  selectedButton: {
    backgroundColor: '#D6CCC2',
  },
  unselectedButton: {
    backgroundColor: '#F2EFED',
  },
  selectedText: {
    fontSize: 20,
    color: '#434240',
  },
  unselectedText: {
    fontSize: 20,
    color: '#B4B4B4',
  },
  modalOverlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalContent: {
    alignItems: 'center',
    width: '90%',
    padding: 10,
    borderRadius: 10,
    backgroundColor: '#FFFFFF',
  },
  modalTitle: {
    marginVertical: 40,
    fontSize: 22,
  },
  modalButtonWrapper: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
  },
  modalButton: {
    flex: 1,
    alignItems: 'center',
    padding: 10,
    borderRadius: 10,
  },
  modalButtonPrimary: {
    backgroundColor: '#F2EAE3',
  },
  modalButtonSecondary: {
    backgroundColor: '#D6CCC2',
  },
  modalButtonText: {
    fontSize: 18,
    color: '#575553',
  },
  modalContentText: {
    fontSize: 22,
    margin: 50,
  },
});

export default LifestyleQuestionScreen;