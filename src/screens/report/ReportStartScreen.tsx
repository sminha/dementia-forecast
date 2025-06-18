import React, { useEffect, useState } from 'react';
import { View, StyleSheet } from 'react-native';
import Modal from 'react-native-modal';
import { useSelector, useDispatch } from 'react-redux';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { AppDispatch, RootState } from '../../redux/store.ts';
import { loadTokens } from '../../redux/actions/authAction.ts';
import { createReport } from '../../redux/slices/reportSlice.ts';
import { sendMessageToChatGPT } from '../../utils/chatgpt.ts';
import FastImage from 'react-native-fast-image';
import CustomText from '../../components/CustomText.tsx';

const ReportStartScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'ReportStart'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();

  const name = useSelector((state: RootState) => state.user.userInfo.name);
  const biometric = useSelector((state: RootState) => state.biometric);
  const [failModal, setFailModal] = useState<boolean>(false);

  const handlePredict = async () => {
    const { accessToken } = await loadTokens();
    if (!accessToken) {
      console.log('로그인 정보가 없습니다.');
      return;
    }

    const biometricInfoInStorage = await AsyncStorage.getItem('biometricInfo');
    if (!biometricInfoInStorage) {
      console.log('생체 정보가 없습니다.');
      return;
    }

    // 1
    // const biometricInfo = JSON.parse(biometricInfoInStorage);
    // 2
    // const biometricInfo = biometric;
    // 3
    // const parsed = JSON.parse(biometricInfoInStorage);
    // const biometricInfo = parsed.biometricInfo;

    // const resultAction = await dispatch(createReport({ token: accessToken, biometricInfo }));
    const resultAction = await dispatch(createReport({ token: accessToken, biometricInfo: biometric }));

    if (createReport.fulfilled.match(resultAction)) {
      const { message } = resultAction.payload;
      console.log('✅', message);

      const prompt = `
      다음은 ${name}님의 라이프스타일 및 생체정보 수치이며, 치매 위험도 예측에 활용됩니다. 각 항목에는 ${name}님의 수치와 치매환자 평균값이 함께 제공됩니다.

      - '작을수록 좋지 않은 항목'과 '클수록 좋지 않은 항목'이 있으며, 각각의 기준에 따라 평균값과의 차이가 큰 항목들을 분석해 주세요.
      - 평균값보다 **현저히 낮은 '작을수록 안 좋은 항목'**과, 평균값보다 **현저히 높은 '클수록 안 좋은 항목'** 중에서 **총 4개 항목을 선정**하세요.
      - 항목별로 개별적으로 나열하지 말고, **줄글 형태로 자연스럽게 이어지는 문장으로 분석 및 권장사항을 5~6문장 이내로 한국어로 작성**해 주세요.
      - 분석 내용은 명확하고 구체적이며, ${name}님이 자신의 상황을 이해하고 개선할 수 있도록 쉽게 설명해 주세요.
      - 수치는 아래 형식으로 제공됩니다: (${name}님 수치 / 치매환자 평균값)

      # 작을수록 안 좋은 항목:
      월 서적 지출, 하루간 활동 칼로리, 하루간 총 사용 칼로리, 총 활동 시간, 저강도 활동 시간, 중강도 활동 시간, 고강도 활동 시간, 하루간 평균 MET, 하루간 비활동 MET, 하루간 저강도 활동 MET, 하루간 중강도 활동 MET, 하루간 고강도 활동 MET, 하루간 걸음수, 하루간 움직인 거리, 총 수면 시간, 깊은 수면 시간, 렘 수면 시간, 수면 효율

      # 클수록 안 좋은 항목:
      월 음주 지출, 월 담배 지출, 비활동 시간, 수면 중 깬 시간, 얕은 수면 시간

      다음은 ${name}님 수치와 평균값입니다:

      1. 월 서적 지출: ${parseInt(resultAction.payload.question_list[12].answer, 10)} / 0.3
      2. 하루간 활동 칼로리: ${Math.floor((resultAction.payload.biometric_data_list[1].biometric_data_value + resultAction.payload.biometric_data_list[29].biometric_data_value + resultAction.payload.biometric_data_list[57].biometric_data_value) / 3)} / 468
      3. 하루간 총 사용 칼로리: ${Math.floor((resultAction.payload.biometric_data_list[2].biometric_data_value + resultAction.payload.biometric_data_list[30].biometric_data_value + resultAction.payload.biometric_data_list[58].biometric_data_value) / 3)} / 2519
      4. 총 활동 시간: ${Math.floor((resultAction.payload.biometric_data_list[14].biometric_data_value + resultAction.payload.biometric_data_list[42].biometric_data_value + resultAction.payload.biometric_data_list[70].biometric_data_value) / 3)} / 341
      5. 저강도 활동 시간: ${Math.floor((resultAction.payload.biometric_data_list[6].biometric_data_value + resultAction.payload.biometric_data_list[34].biometric_data_value + resultAction.payload.biometric_data_list[62].biometric_data_value) / 3)} / 285
      6. 중강도 활동 시간: ${Math.floor((resultAction.payload.biometric_data_list[7].biometric_data_value + resultAction.payload.biometric_data_list[35].biometric_data_value + resultAction.payload.biometric_data_list[63].biometric_data_value) / 3)} / 53
      7. 고강도 활동 시간: ${Math.floor((resultAction.payload.biometric_data_list[4].biometric_data_value + resultAction.payload.biometric_data_list[32].biometric_data_value + resultAction.payload.biometric_data_list[60].biometric_data_value) / 3)} / 3
      8. 하루간 평균 MET: ${Math.floor((resultAction.payload.biometric_data_list[0].biometric_data_value + resultAction.payload.biometric_data_list[28].biometric_data_value + resultAction.payload.biometric_data_list[56].biometric_data_value) / 3)} / 1.5
      9. 하루간 비활동 MET: ${Math.floor((resultAction.payload.biometric_data_list[9].biometric_data_value + resultAction.payload.biometric_data_list[37].biometric_data_value + resultAction.payload.biometric_data_list[65].biometric_data_value) / 3)} / 7
      10. 하루간 저강도 활동 MET: ${Math.floor((resultAction.payload.biometric_data_list[10].biometric_data_value + resultAction.payload.biometric_data_list[38].biometric_data_value + resultAction.payload.biometric_data_list[66].biometric_data_value) / 3)} / 192
      11. 하루간 중강도 활동 MET: ${Math.floor((resultAction.payload.biometric_data_list[11].biometric_data_value + resultAction.payload.biometric_data_list[39].biometric_data_value + resultAction.payload.biometric_data_list[67].biometric_data_value) / 3)} / 169
      12. 하루간 고강도 활동 MET: ${Math.floor((resultAction.payload.biometric_data_list[8].biometric_data_value + resultAction.payload.biometric_data_list[36].biometric_data_value + resultAction.payload.biometric_data_list[64].biometric_data_value) / 3)} / 20
      13. 하루간 걸음수: ${Math.floor((resultAction.payload.biometric_data_list[13].biometric_data_value + resultAction.payload.biometric_data_list[41].biometric_data_value + resultAction.payload.biometric_data_list[69].biometric_data_value) / 3)} / 10772
      14. 하루간 움직인 거리: ${Math.floor((resultAction.payload.biometric_data_list[3].biometric_data_value + resultAction.payload.biometric_data_list[31].biometric_data_value + resultAction.payload.biometric_data_list[59].biometric_data_value) / 3)} / 8773
      15. 총 수면 시간: ${Math.floor((resultAction.payload.biometric_data_list[27].biometric_data_value + resultAction.payload.biometric_data_list[55].biometric_data_value + resultAction.payload.biometric_data_list[83].biometric_data_value) / 3 / 60 / 60)} / 6
      16. 깊은 수면 시간: ${Math.floor((resultAction.payload.biometric_data_list[16].biometric_data_value + resultAction.payload.biometric_data_list[44].biometric_data_value + resultAction.payload.biometric_data_list[72].biometric_data_value) / 3 / 60 / 60)} / 1
      17. 렘 수면 시간: ${Math.floor((resultAction.payload.biometric_data_list[25].biometric_data_value + resultAction.payload.biometric_data_list[53].biometric_data_value + resultAction.payload.biometric_data_list[81].biometric_data_value) / 3 / 60 / 60)} / 1
      18. 수면 효율: ${Math.floor((resultAction.payload.biometric_data_list[18].biometric_data_value + resultAction.payload.biometric_data_list[46].biometric_data_value + resultAction.payload.biometric_data_list[74].biometric_data_value) / 3)} / 81
      19. 월 음주 지출: ${parseInt(resultAction.payload.question_list[8].answer, 10)} / 0.7
      20. 월 담배 지출: ${parseInt(resultAction.payload.question_list[9].answer, 10)} / 1.5
      21. 비활동 시간: ${Math.floor((resultAction.payload.biometric_data_list[5].biometric_data_value + resultAction.payload.biometric_data_list[33].biometric_data_value + resultAction.payload.biometric_data_list[61].biometric_data_value) / 3)} / 502
      22. 수면 중 깬 시간: ${Math.floor((resultAction.payload.biometric_data_list[15].biometric_data_value + resultAction.payload.biometric_data_list[43].biometric_data_value + resultAction.payload.biometric_data_list[71].biometric_data_value) / 3 / 60 / 60)} / 1
      23. 얕은 수면 시간: ${Math.floor((resultAction.payload.biometric_data_list[21].biometric_data_value + resultAction.payload.biometric_data_list[49].biometric_data_value + resultAction.payload.biometric_data_list[77].biometric_data_value) / 3 / 60 / 60)} / 4
      `;

      const response = await sendMessageToChatGPT(prompt);
      // console.log(response);

      // const response = '홍길동님의 라이프스타일 및 생체정보를 분석해보면 몇 가지 주목할 만한 점이 있습니다. 먼저, 월 음주 지출과 월 담배 지출이 평균값보다 현저히 높은 것으로 나타납니다. 이러한 습관은 건강에 해로울 수 있으므로 적절한 관리가 필요합니다.'; // FOR TEST

      let isNavigated = false;

      // navigation.replace('ReportResult', { response });
      setTimeout(() => {
        if (!isNavigated) {
          setFailModal(false);
        }
      }, 5000);

      navigation.replace('ReportResult', { type: 'reportStart', response: response });
      isNavigated = true;

      AsyncStorage.removeItem('biometricInfo');

      const today = new Date();
      const yyyy = today.getFullYear();
      const mm = String(today.getMonth() + 1).padStart(2, '0');
      const dd = String(today.getDate()).padStart(2, '0');
      const newCreateDate = `${yyyy}${mm}${dd}`;

      const data = await AsyncStorage.getItem('createDate');
      const parsedArray = data ? JSON.parse(data) : [];
      parsedArray.push(parseInt(newCreateDate, 10));
      await AsyncStorage.setItem('createDate', JSON.stringify(parsedArray));
    } else if (createReport.rejected.match(resultAction)) {
      const errorMessage = (resultAction.payload as any)?.message || '레포트 생성 실패';
      console.log('❌', errorMessage);

      setFailModal(true);

      const timer = setTimeout(() => {
        setFailModal(false);
        navigation.replace('Home');
      }, 3000);

      return () => clearTimeout(timer);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      handlePredict();
    }, 3000);

    return () => clearTimeout(timer);
  });

  return (
    <View style={styles.container}>
      <View style={styles.textContainer}>
        <CustomText style={styles.title}>알려주신 정보로{'\n'}치매 위험도를 진단하는 중이에요.{'\n\n'}잠시만 기다려주세요!</CustomText>
      </View>

      <View style={styles.imageContainer}>
        <FastImage source={require('../../assets/images/report_start.gif')} style={styles.image} />
      </View>

      <Modal
        isVisible={failModal}
        backdropColor="rgb(69, 69, 69)"
        backdropOpacity={0.3}
        animationIn="fadeIn"
        animationOut="fadeOut"
        style={styles.modalOverlay}
      >
        <View style={styles.modalContent}>
          <CustomText style={styles.modalTitle}>치매 위험도 진단에 실패했어요.</CustomText>
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
  textContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  title: {
    textAlign: 'center',
    fontSize: 24,
  },
  imageContainer: {
    flex: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: 130,
    height: 130,
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
    fontSize: 21,
  },
  modalText: {
    // marginVertical: 40,
    marginBottom: 40,
    fontSize: 17,
    color: '#575553',
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

export default ReportStartScreen;