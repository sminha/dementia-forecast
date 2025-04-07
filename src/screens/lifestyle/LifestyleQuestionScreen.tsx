import React, { useState } from 'react';
import { View, SafeAreaView, TouchableOpacity, Modal, TouchableWithoutFeedback, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const LifestyleQuestionScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'LifestyleQuestion'>;
  const navigation = useNavigation<Navigation>();

  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);

  const [modalVisible, setModalVisible] = useState(false);

  const lifestyleQuestion = [
    {
      question: '한 달 소득이 얼마인가요?',
      options: ['100만원 이하', '200만원 이하', '300만원 이하', '300만원 초과'],
    },
    {
      question: '한 달 동안 음주로 지출한 비용이 얼마인가요?',
      options: ['5만원 이하', '10만원 이하', '15만원 이하', '20만원 초과'],
    },
  ];

  const currentQuestion = lifestyleQuestion[currentQuestionIndex];
  const selectedOption = selectedOptions[currentQuestionIndex] || null;

  return (
    <View style={styles.container}>
      <SafeAreaView>
        <View style={styles.headerContainer}>
          <TouchableOpacity onPress={() => {
            if (currentQuestionIndex !== 0) {
              setCurrentQuestionIndex((prev) => prev - 1);
            } else {
              setModalVisible(true)
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
              navigation.navigate('LifestyleComplete');
            }
          }}
        >
          <CustomText style={selectedOption ? styles.selectedText : styles.unselectedText}>
            {currentQuestionIndex !== lifestyleQuestion.length - 1 ? '다음' : '완료'}
          </CustomText>
        </TouchableOpacity>
      </View>

      <Modal
        transparent={true}
        animationType="fade"
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <TouchableWithoutFeedback onPress={() => setModalVisible(false)}>
          <View style={styles.modalOverlay}>
            <TouchableWithoutFeedback>
              <View style={styles.modalContent}>
                <CustomText style={styles.modalTitle}>라이프스타일 입력을 그만할까요?</CustomText>

                <View style={styles.modalButtonWrapper}>
                  <TouchableOpacity style={[styles.modalButton, styles.modalButtonPrimary]} onPress={() => navigation.navigate('Home')}>
                    <CustomText style={styles.modalButtonText}>네, 그만할래요</CustomText>
                  </TouchableOpacity>

                  <TouchableOpacity style={[styles.modalButton, styles.modalButtonSecondary]} onPress={() => setModalVisible(false)}>
                    <CustomText style={styles.modalButtonText}>아니요, 계속 할래요</CustomText>
                  </TouchableOpacity>
                </View>
              </View>
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
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
    paddingTop: 16,
    paddingLeft: 10,
  },
  contentContainer: {
    paddingTop: 50,
    paddingHorizontal: 10,
  },
  questionText: {
    minHeight: 60,
    marginBottom: 50,
    fontSize: 16,
  },
  optionContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
  },
  optionText: {
    paddingBottom: 16,
    fontSize: 14,
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
    fontSize: 14,
    color: '#434240',
  },
  unselectedText: {
    fontSize: 14,
    color: '#B4B4B4',
  },
  modalOverlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(178, 178, 178, 0.3)',
  },
  modalContent: {
    alignItems: 'center',
    width: '90%',
    padding: 10,
    borderRadius: 10,
    backgroundColor: '#FFFFFF',
  },
  modalTitle: {
    marginVertical: 30,
    fontSize: 14,
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
    fontSize: 10,
    color: '#575553',
  },
});

export default LifestyleQuestionScreen;