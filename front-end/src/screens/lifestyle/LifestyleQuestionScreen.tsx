import React, { useState, useRef, useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, StyleSheet, Pressable } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import Modal from 'react-native-modal';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
// import { saveLifestyle } from '../../api/lifestyleApi.ts';
import { loadTokens } from '../../redux/actions/authAction.ts';
import { AppDispatch, RootState } from '../../redux/store.ts';
import { saveLifestyle, clearSaveResult } from '../../redux/slices/lifestyleSlice.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';
import { TextInput } from 'react-native-gesture-handler';

import SelectOptionModal from '../../components/SelectOptionModal.tsx';

const LifestyleQuestionScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();

  const userInfo = useSelector((state: RootState) => state.user.userInfo);
  const saveResult = useSelector((state: RootState) => state.lifestyle.saveResult);
  const isLoading = useSelector((state: RootState) => state.lifestyle.isLoading);

  const [answers, setAnswers] = useState<string[]>([]);
  const [currentInputIndex, setCurrentInputIndex] = useState<number>(0);
  const [spouse, setSpouse] = useState<boolean | null>(null);
  const [pet, setPet] = useState<boolean | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [backConfirmModal, setBackConfirmModal] = useState<boolean>(false);
  const [saveFailModal, setSaveFailModal] = useState<boolean>(false);
  const [authFailModal, setAuthFailModal] = useState<boolean>(false);

  const inputRefs = useRef<Array<React.RefObject<TextInput | null>>>([]);
  const inputTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const lifestyleQuestions = [
    '가구원이 몇 명인가요?',
    '배우자가 있나요?',
    '월 소득이 얼마인가요?',
    '월 지출이 얼마인가요?',
    '한 달 동안 음주에 지출한 비용이 얼마인가요?',
    '한 달 동안 담배에 지출한 비용이 얼마인가요?',
    '한 달 동안 서적에 지출한 비용이 얼마인가요?',
    '한 달 동안 복지시설에 지출한 비용이 얼마인가요?',
    '한 달 의료비가 얼마인가요?',
    '한 달 보험비가 얼마인가요?',
    '애완동물이 있나요?',
    '현재 가정의 형태가 어떻게 되나요?',
  ];

  const lifestyleQuestionTopics = [
    '가구원 수',
    '배우자 유무',
    '월 소득',
    '월 지출',
    '월 음주 지출',
    '월 담배 지출',
    '월 서적 지출',
    '월 복지시설 지출',
    '월 의료비',
    '월 보험비',
    '애완동물 유무',
    '가정 형태',
  ];

  const householdOptions = ['일반 가정', '결손 가정', '나홀로 가정', '저소득 가정', '저소득 결손 가정', '저소득 나홀로 가정'];

  useEffect(() => {
    if (inputRefs.current.length !== lifestyleQuestions.length) {
      lifestyleQuestions.forEach((_, i) => {
        if (!inputRefs.current[i]) {
          inputRefs.current[i] = React.createRef<TextInput>();
        }
      });
    }

    const timeout = setTimeout(() => {
      inputRefs.current[currentInputIndex]?.current?.focus();
    }, 10);

    return () => clearTimeout(timeout);
  }, [currentInputIndex]);

  useEffect(() => {
    if (authFailModal) {
      const timer = setTimeout(() => {
        navigation.navigate('Login');
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [navigation, authFailModal]);

  useEffect(() => {
    // if (updateResult?.statusCode === 200) {
    if (saveResult?.message === '라이프스타일 저장 및 수정 성공') {
      // dispatch(setUserInfo({ field: 'birthdate', value: isoBirthdate }));
      navigation.replace('LifestyleComplete');
      dispatch(clearSaveResult());
    } else if (saveResult?.message === '비밀키나 토큰 포맷이 맞지 않습니다.') {
      dispatch(clearSaveResult());
      setAuthFailModal(true);
    }
    // else {
    //   setSaveFailModal(true);
    // }
  }, [saveResult, navigation, dispatch]);

  const handleInputChange = (text: string, index: number) => {
    const updated = [...answers];
    updated[index] = text;
    setAnswers(updated);

    if (inputTimeoutRef.current) {
      clearTimeout(inputTimeoutRef.current);
    }

    if (!text || text.trim() === '') {
      return;
    }

    inputTimeoutRef.current = setTimeout(() => {
      if (index === currentInputIndex && index < lifestyleQuestions.length - 1) {
        setCurrentInputIndex(index + 1);
      }
    }, 1000);
  };

  // console.log(answers);
  // console.log('라이프스타일 채우기 전 사용자 정보:', userInfo);

  const handleSubmit = async () => {
    const genderValue = userInfo.gender === '남' ? '0' : '1';
    const year = userInfo.birthdate.slice(0, 4);
    const month = userInfo.birthdate.slice(5, 7);
    const day = userInfo.birthdate.slice(8, 10);

    const baseInfo = [
      { question_id: 1, answer: genderValue },
      { question_id: 2, answer: year },
      { question_id: 3, answer: month },
      { question_id: 4, answer: day },
    ];

    const lifestyleInfo = lifestyleQuestions.map((_, index) => ({
      question_id: index + 5,
      answer: answers[index],
    }));

    const questionList = [...baseInfo, ...lifestyleInfo];

    // const questionList = lifestyleQuestions.map((question, index) => ({
    //   question_id: index + 1,
    //   answer: answers[index],
    // }));

    // console.log('전달할 인자:', questionList);

    const { accessToken } = await loadTokens();
    if (!accessToken) {
      console.log('로그인 정보가 없습니다.');
      return;
    }

    dispatch(saveLifestyle({ token: accessToken, questionList: questionList }));

    navigation.navigate('LifestyleComplete');
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          <TouchableOpacity style={styles.backContainer} onPress={() => setBackConfirmModal(true)}>
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <CustomText style={styles.sectionTitle}>{lifestyleQuestions[currentInputIndex]}</CustomText>

          {[...lifestyleQuestionTopics]
          .map((topic, index) => ({ topic, index }))
          .filter(({ index }) => index <= currentInputIndex)
          .reverse()
          .map(({ topic, index }) => (
            <View key={index} style={styles.row}>
              <CustomText style={[styles.sectionText, focusedField === topic && styles.focusedSectionText]}>{topic}</CustomText>
              {topic === '가구원 수' ? (
                <View>
                  <TextInput
                    style={[styles.inputField, focusedField === topic && styles.focusedInputField]}
                    value={answers[index]}
                    onFocus={() => setFocusedField(topic)}
                    onBlur={() => setFocusedField(null)}
                    onChangeText={(text) => handleInputChange(text, index)}
                    keyboardType="numeric"
                  />
                </View>
              ) : topic === '배우자 유무' ? (
                <View style={styles.genderButtonContainer}>
                  <TouchableOpacity
                    style={[styles.genderButtonLeft, spouse === true && styles.genderButtonSelected]}
                    onPress={() => {
                      setSpouse(true);
                      setFocusedField(topic);
                      handleInputChange('있음', index);
                    }}
                  >
                    <CustomText style={[styles.genderButtonText, spouse === true && styles.genderButtonTextSelected]}>있음</CustomText>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[styles.genderButtonRight, spouse === false && styles.genderButtonSelected]}
                    onPress={() => {
                      setSpouse(false);
                      setFocusedField(topic);
                      handleInputChange('없음', index);
                    }}
                  >
                    <CustomText style={[styles.genderButtonText, spouse === false && styles.genderButtonTextSelected]}>없음</CustomText>
                  </TouchableOpacity>
                </View>
              ) : topic === '애완동물 유무' ? (
                <View style={styles.genderButtonContainer}>
                  <TouchableOpacity
                    style={[styles.genderButtonLeft, pet === true && styles.genderButtonSelected]}
                    onPress={() => {
                      setPet(true);
                      setFocusedField(topic);
                      handleInputChange('있음', index);
                    }}
                  >
                    <CustomText style={[styles.genderButtonText, pet === true && styles.genderButtonTextSelected]}>있음</CustomText>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[styles.genderButtonRight, pet === false && styles.genderButtonSelected]}
                    onPress={() => {
                      setPet(false);
                      setFocusedField(topic);
                      handleInputChange('없음', index);
                    }}
                  >
                    <CustomText style={[styles.genderButtonText, pet === false && styles.genderButtonTextSelected]}>없음</CustomText>
                  </TouchableOpacity>
                </View>
              ) : topic === '가정 형태' ? (
                <View>
                  <Pressable
                    onPress={() => setModalVisible(true)}
                    style={styles.inputField}
                  >
                    <CustomText style={selectedOption ? styles.inputFieldText : styles.inputFieldContent}>{selectedOption ?? '가정 형태를 선택하세요.'}</CustomText>
                  </Pressable>

                  <SelectOptionModal
                    visible={modalVisible}
                    options={householdOptions}
                    onSelect={(option) => {
                      setSelectedOption(option);
                      setModalVisible(false);
                      handleInputChange(option, index);
                    }}
                    onClose={() => setModalVisible(false)}
                  />
                </View>
              ) : (
                <View>
                  <TextInput
                    style={[styles.inputField, focusedField === topic && styles.focusedInputField]}
                    value={answers[index]}
                    onFocus={() => setFocusedField(topic)}
                    onBlur={() => setFocusedField(null)}
                    onChangeText={(text) => handleInputChange(text, index)}
                    keyboardType="numeric"
                    placeholder="만 원 단위로 입력하세요."
                    ref={inputRefs.current[index]}
                  />
                </View>
              )}
            </View>))
          }
        </View>
      </ScrollView>

      <View style={styles.submitButtonWrapper}>
        {answers.length === lifestyleQuestions.length && answers.every(answer => answer.trim() !== '') ? (
          <TouchableOpacity style={styles.submitButtonEnabled} onPress={handleSubmit}>
            <CustomText style={styles.submitButtonTextEnabled}>{isLoading ? '저장 중...' : '완료'}</CustomText>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={styles.submitButtonDisabled} onPress={handleSubmit}>
            <CustomText style={styles.submitButtonTextDisabled}>{isLoading ? '저장 중...' : '완료'}</CustomText>
          </TouchableOpacity>
        )}
      </View>

      <Modal
        isVisible={backConfirmModal}
        onBackdropPress={() => setBackConfirmModal(false)}
        onBackButtonPress={() => setBackConfirmModal(false)}
        backdropColor="rgb(69, 69, 69)"
        backdropOpacity={0.3}
        animationIn="fadeIn"
        animationOut="fadeOut"
        style={styles.modalOverlay}
      >
        <View style={styles.modalContent}>
          <CustomText style={styles.modalTitle}>라이프스타일 입력을 그만할까요?</CustomText>

          <View style={styles.modalButtonWrapper}>
            <TouchableOpacity style={[styles.modalButton, styles.modalButtonPrimary]} onPress={() => navigation.goBack()}>
              <CustomText style={styles.modalButtonText}>네, 그만할래요</CustomText>
            </TouchableOpacity>

            <TouchableOpacity style={[styles.modalButton, styles.modalButtonSecondary]} onPress={() => setBackConfirmModal(false)}>
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
          <CustomText style={styles.modalContentText}>로그인 후 진행해주세요.</CustomText>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollContent: {
    paddingVertical: 10,
    paddingHorizontal: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    position: 'relative',
    height: 50,
  },
  backContainer: {
    paddingHorizontal: 4,
    zIndex: 1,
  },
  titleWrapper: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loginText: {
    fontSize: 24,
  },
  section: {
    marginTop: 20,
  },
  sectionSecondary: {
    marginTop: 30,
  },
  sectionTitle: {
    fontSize: 24,
    marginBottom: 8,
  },
  sectionText: {
    marginTop: 30,
    fontSize: 18,
    color: '#202020',
  },
  focusedSectionText: {
    marginTop: 30,
    fontSize: 18,
    color: '#9F8473',
  },
  row: {
    marginHorizontal: 5,
  },
  inputField: {
    width: '100%',
    // width: 100,
    paddingRight: 35,
    marginTop: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    fontSize: 18,
  },
  inputFieldText: {
    marginVertical: 10,
    fontSize: 18,
  },
  inputFieldContent: {
    marginVertical: 10,
    fontSize: 18,
    color: '#A9A9A9',
  },
  focusedInputField: {
    width: '100%',
    // width: 100,
    paddingRight: 35,
    marginTop: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#9F8473',
    fontSize: 18,
  },
  genderButtonContainer: {
    flexDirection: 'row',
  },
  genderButtonLeft: {
    flex: 50,
    alignItems: 'center',
    padding: 10,
    marginTop: 20,
    marginRight: 7,
    borderWidth: 1,
    borderColor: '#B4B4B4',
  },
  genderButtonRight: {
    flex: 50,
    alignItems: 'center',
    padding: 10,
    marginTop: 20,
    marginLeft: 7,
    borderWidth: 1,
    borderColor: '#B4B4B4',
  },
  genderButtonSelected: {
    borderColor: '#D6CCC2',
    backgroundColor: '#D6CCC2',
  },
  genderButtonText: {
    fontSize: 18,
    color: '#6C6C6B',
  },
  genderButtonTextSelected: {
    color: '#575553',
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
  submitButtonWrapper: {
    paddingVertical: 10,
    paddingHorizontal: 16,
  },
  submitButtonEnabled: {
    alignItems: 'center',
    padding: 15,
    marginTop: 10,
    marginBottom: 5,
    borderRadius: 5,
    backgroundColor: '#D6CCC2',
  },
  submitButtonTextEnabled: {
    fontSize: 20,
    color: '#575553',
  },
    submitButtonDisabled: {
    alignItems: 'center',
    padding: 15,
    marginTop: 10,
    marginBottom: 5,
    borderRadius: 5,
    backgroundColor: '#F2EFED',
  },
  submitButtonTextDisabled: {
    fontSize: 20,
    color: '#B4B4B4',
  },
});

export default LifestyleQuestionScreen;