import React, { useState, useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, TextInput, Pressable, StyleSheet } from 'react-native';
import Modal from 'react-native-modal';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigation, RouteProp, useRoute } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../../types/navigationTypes.ts';
import { saveLifestyle } from '../../../api/lifestyleApi.ts';
import { HouseholdType, setLifestyleInfo, LifestyleInfo } from '../../../redux/slices/lifestyleSlice.ts';
import { AppDispatch, RootState } from '../../../redux/store.ts';
import CustomText from '../../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';
import SelectOptionModal from '../../../components/SelectOptionModal.tsx';

const LifestyleEditScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();
  type LifestyleEditRouteProp = RouteProp<RootStackParamList, 'LifestyleEdit'>;
  const route = useRoute<LifestyleEditRouteProp>();
  const { topic } = route.params as { topic: string };

  const dispatch = useDispatch<AppDispatch>();
  const userInfo = useSelector((state: RootState) => state.user.userInfo);
  const lifestyleInfo = useSelector((state: RootState) => state.lifestyle.lifestyleInfo);
  const isLoading = useSelector((state: RootState) => state.lifestyle.isLoading);

  const [lifestyle, setLifestyle] = useState<string>('');
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [saveFailModal, setSaveFailModal] = useState<boolean>(false);
  const [authFailModal, setAuthFailModal] = useState<boolean>(false);

  const householdOptions = ['일반 가정', '결손 가정', '나홀로 가정', '저소득 가정', '저소득 결손 가정', '저소득 나홀로 가정'];

  useEffect(() => {
    if (authFailModal) {
      const timer = setTimeout(() => {
        navigation.navigate('Login');
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [navigation, authFailModal]);

  const handleFocus = (field: string) => {
    setFocusedField(field);
  };

  const handleBlur = () => {
    setFocusedField(null);
  };

  const isFormValid = () => {
    return lifestyle.trim() !== ''/* && lifestyle !== String(lifestyleInfo[topicKeyMap[topic]])*/;
  };

  // const handleUpdate = async () => {
  //   const topicKeyMap: { [key: string]: keyof LifestyleInfo } = {
  //     '가구원 수': 'householdSize',
  //     '배우자 유무': 'hasSpouse',
  //     '월 소득': 'income',
  //     '월 지출': 'expenses',
  //     '월 음주 지출': 'alcoholExpense',
  //     '월 담배 지출': 'tobaccoExpense',
  //     '월 서적 지출': 'bookExpense',
  //     '월 복지시설 지출': 'welfareExpense',
  //     '월 의료비': 'medicalExpense',
  //     '월 보험비': 'insuranceExpense',
  //     '애완동물 유무': 'hasPet',
  //     '가정 형태': 'householdType',
  //   };

  //   const questionList = Object.entries(lifestyleInfo).map(([key, value], index) => {
  //     const currentKey = topicKeyMap[topic];

  //     if (key === currentKey) {
  //       return {
  //         question_id: index + 1,
  //         answer: lifestyle,
  //       };
  //     }

  //     return {
  //       question_id: index + 1,
  //       answer: typeof value === 'boolean' ? (value ? '있음' : '없음') : String(value),
  //     };
  //   });


  //   try {
  //     const response = await saveLifestyle(questionList);

  //     console.log(questionList);
  //     console.log(response);

  //     if (response.message === '라이프스타일 저장 성공') {
  //       if (topicKeyMap[topic] === 'hasSpouse' || topicKeyMap[topic] === 'hasPet') {
  //         if (lifestyle === '있음') {
  //           dispatch(setLifestyleInfo({ field: topicKeyMap[topic], value: true }));
  //         } else {
  //           dispatch(setLifestyleInfo({ field: topicKeyMap[topic], value: false }));
  //         }
  //       } else if (topicKeyMap[topic] === 'householdType') {
  //         dispatch(setLifestyleInfo({ field: topicKeyMap[topic], value: lifestyle as HouseholdType }));
  //       } else {
  //         dispatch(setLifestyleInfo({ field: topicKeyMap[topic], value: parseInt(lifestyle, 10) }));
  //       }
  //       // dispatch(setLifestyleInfo({ field: topicKeyMap[topic], value: lifestyle }));
  //       navigation.goBack();
  //     } else if (response.message === '라이프스타일 저장 실패') {
  //       setSaveFailModal(true);
  //     } else {
  //       setAuthFailModal(true);
  //     }
  //   } catch (error) {
  //     setSaveFailModal(true);
  //   }
  // };

  const handleUpdate = async () => {
    const topicKeyMap: { [key: string]: keyof LifestyleInfo } = {
      '가구원 수': 'householdSize',
      '배우자 유무': 'hasSpouse',
      '월 소득': 'income',
      '월 지출': 'expenses',
      '월 음주 지출': 'alcoholExpense',
      '월 담배 지출': 'tobaccoExpense',
      '월 서적 지출': 'bookExpense',
      '월 복지시설 지출': 'welfareExpense',
      '월 의료비': 'medicalExpense',
      '월 보험비': 'insuranceExpense',
      '애완동물 유무': 'hasPet',
      '가정 형태': 'householdType',
    };

    // userInfo에서 기본 정보 추출
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

    // lifestyleInfo 기반 항목 구성
    const lifestyleEntries = Object.entries(lifestyleInfo);
    const lifestyleInfoList = lifestyleEntries.map(([key, value], index) => {
      const currentKey = topicKeyMap[topic];

      if (key === currentKey) {
        return {
          question_id: index + 5,
          answer: lifestyle,
        };
      }

      return {
        question_id: index + 5,
        answer: typeof value === 'boolean' ? (value ? '있음' : '없음') : String(value),
      };
    });

    const questionList = [...baseInfo, ...lifestyleInfoList];

    try {
      const response = await saveLifestyle(questionList);

      // console.log(questionList);
      // console.log(response);

      if (response.message === '라이프스타일 저장 및 수정 성공') {
        const targetKey = topicKeyMap[topic];

        if (targetKey === 'hasSpouse' || targetKey === 'hasPet') {
          dispatch(setLifestyleInfo({ field: targetKey, value: lifestyle === '있음' }));
        } else if (targetKey === 'householdType') {
          dispatch(setLifestyleInfo({ field: targetKey, value: lifestyle as HouseholdType }));
        } else {
          dispatch(setLifestyleInfo({ field: targetKey, value: parseInt(lifestyle, 10) }));
        }

        navigation.goBack();
      } else if (response.message === '비밀키나 토큰 포맷이 맞지 않습니다.') {
        setAuthFailModal(true);
      }
      // else {
      //   setSaveFailModal(true);
      // }
    } catch (error) {
      setSaveFailModal(true);
    }
  };


  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          <TouchableOpacity style={styles.backContainer} onPress={() => navigation.goBack()}>
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <View style={styles.title}>
            <TouchableOpacity style={styles.titleRow}>
              <CustomText style={styles.titleText}>{topic} 수정하기</CustomText>
            </TouchableOpacity>
          </View>
        </View>

        {topic === '가구원 수' ? (
          <View style={styles.inputGroup}>
            <CustomText style={[styles.labelText, focusedField === 'lifestyle' && styles.labelTextFocused]}>{topic}</CustomText>
            <View style={styles.inputFieldWrapper}>
              <TextInput
                style={[styles.inputField, focusedField === 'lifestyle' && styles.inputFieldFocused]}
                onFocus={() => handleFocus('lifestyle')}
                onBlur={() => handleBlur()}
                onChangeText={(text) => {
                  setLifestyle(text);
                  // dispatch(setUserInfo({ field: 'phone', value: text }));
                }}
                value={lifestyle}
                keyboardType="numeric"
              />
              {lifestyle !== '' &&
              <TouchableOpacity onPress={() => setLifestyle('')} style={styles.clearButton}>
                <Icon name="close-circle" size={20} color="#B4B4B4" />
              </TouchableOpacity>
              }
            </View>
          </View>
        ) : topic === '배우자 유무' || topic === '애완동물 유무' ? (
          <View style={styles.rowWithoutMatch}>
            <CustomText style={[styles.label, /*focusedField === 'gender' &&*/ styles.focusedLabel]}>{topic}</CustomText>
            <View style={styles.genderButtonContainer}>
              <TouchableOpacity
                style={[styles.genderButtonLeft, lifestyle === '있음' && styles.genderButtonSelected]}
                onPress={() => setLifestyle('있음') /*dispatch(setUserInfo({ field: 'gender', value: '남' }))*/}
              >
                <CustomText style={[styles.genderButtonText, lifestyle === '있음' && styles.genderButtonTextSelected]}>있음</CustomText>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.genderButtonRight, lifestyle === '없음' && styles.genderButtonSelected]}
                onPress={() => setLifestyle('없음') /*dispatch(setUserInfo({ field: 'gender', value: '여' }))*/}
              >
                <CustomText style={[styles.genderButtonText, lifestyle === '없음' && styles.genderButtonTextSelected]}>없음</CustomText>
              </TouchableOpacity>
            </View>
          </View>
        ) : topic === '가정 형태' ? (
          <View style={styles.pressableWrapper}>
            <CustomText style={[styles.labelText, focusedField === 'lifestyle' && styles.labelTextFocused]}>{topic}</CustomText>
            <Pressable
              onPress={() => {
                setModalVisible(true);
                setFocusedField('lifestyle');
              }}
              style={[styles.inputField, focusedField === 'lifestyle' && styles.inputFieldFocused]}
            >
              <CustomText style={selectedOption ? styles.inputFieldText : styles.inputFieldContent}>{selectedOption ?? '가정 형태를 선택하세요.'}</CustomText>
            </Pressable>

            <SelectOptionModal
              visible={modalVisible}
              options={householdOptions}
              onSelect={(option) => {
                setSelectedOption(option);
                setModalVisible(false);
                setLifestyle(option);
                setFocusedField('');
                // handleInputChange(option, index);
              }}
              onClose={() => setModalVisible(false)}
            />
          </View>
        ) : (
          <View style={styles.inputGroup}>
            <CustomText style={[styles.labelText, focusedField === 'lifestyle' && styles.labelTextFocused]}>{topic}</CustomText>
            <View style={styles.inputFieldWrapper}>
              <TextInput
                style={[styles.inputField, focusedField === 'lifestyle' && styles.inputFieldFocused]}
                onFocus={() => handleFocus('lifestyle')}
                onBlur={() => handleBlur()}
                onChangeText={(text) => {
                  setLifestyle(text);
                  // dispatch(setUserInfo({ field: 'phone', value: text }));
                }}
                value={lifestyle}
                keyboardType="numeric"
                placeholder="만 원 단위로 입력하세요."
              />
              {lifestyle !== '' &&
              <TouchableOpacity onPress={() => setLifestyle('')} style={styles.clearButton}>
                <Icon name="close-circle" size={20} color="#B4B4B4" />
              </TouchableOpacity>
              }
            </View>
          </View>
        )}

        <View>
          <TouchableOpacity onPress={handleUpdate} style={[styles.actionButton, isFormValid() ? styles.actionButtonEnabled : styles.actionButtonDisabled]} disabled={!isFormValid()}>
            <CustomText style={[styles.actionButtonText, isFormValid() ? styles.actionButtonTextEnabled : styles.actionButtonTextDisabled]}>
              수정 완료
            </CustomText>
          </TouchableOpacity>
        </View>
      </ScrollView>

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
                handleUpdate();
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
  section: {
    marginTop: 20,
  },
  title: {
    paddingHorizontal: 5,
    paddingTop: 10,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    // marginBottom: 20,
    marginBottom: 30,
  },
  titleText: {
    fontSize: 24,
  },
  inputGroup: {
    marginBottom: 40,
    marginHorizontal: 5,
  },
  labelText: {
    fontSize: 18,
    color: '#202020',
  },
  labelTextFocused: {
    color: '#9F8473',
  },
  inputFieldWrapper: {
    // position: 'relative',
    flexDirection: 'column',
    // alignItems: 'center',
  },
  clearButton: {
    position: 'absolute',
    top: '40%',
    right: 1,
    transform: [{ translateY: -8 }],
  },
  inputField: {
    width: '100%',
    paddingRight: 35,
    marginTop: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    fontSize: 18,
  },
  inputFieldFocused: {
    borderBottomColor: '#9F8473',
  },
  pressableWrapper: {
    marginBottom: 40,
    marginHorizontal: 5,
  },
  confirmContainer: {
    height: 20,
  },
  confirmText: {
    marginTop: 5,
    color: '#D64747',
  },
  confirmTextHidden: {
    marginTop: 5,
    color: 'transparent',
  },
  actionButton: {
    alignItems: 'center',
    padding: 15,
    marginTop: 10,
    marginBottom: 5,
    marginHorizontal: 5,
    borderRadius: 5,
  },
  actionButtonEnabled: {
    backgroundColor: '#D6CCC2',
  },
  actionButtonDisabled: {
    backgroundColor: '#F2EFED',
  },
  actionButtonText: {
    fontSize: 20,
  },
  actionButtonTextEnabled: {
    color: '#575553',
  },
  actionButtonTextDisabled: {
    color: '#B4B4B4',
  },


  rowWithoutMatch: {
    marginBottom: 40,
    marginHorizontal: 5,
  },
  label: {
    fontSize: 18,
    color: '#202020',
  },
  focusedLabel: {
    color: '#9F8473',
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
    inputFieldText: {
    marginVertical: 10,
    fontSize: 18,
  },
  inputFieldContent: {
    marginVertical: 10,
    fontSize: 18,
    color: '#A9A9A9',
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

export default LifestyleEditScreen;