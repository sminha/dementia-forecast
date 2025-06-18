import React, { useState, useEffect } from 'react';
import { View, SafeAreaView, TouchableOpacity, Image, ScrollView, ActivityIndicator, StyleSheet } from 'react-native';
import { useSelector } from 'react-redux';
import { useNavigation, RouteProp, useRoute } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { RootState } from '../../redux/store.ts';
import { sendMessageToChatGPT } from '../../utils/chatgpt.ts';
import { AnimatedCircularProgress } from 'react-native-circular-progress';
import { ProgressBar } from 'react-native-paper';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const ReportResultScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'ReportResult'>;
  const navigation = useNavigation<Navigation>();

  const route = useRoute<RouteProp<RootStackParamList, 'ReportResult'>>();

  const name = useSelector((state: RootState) => state.user.userInfo.name);
  const reportResult = useSelector((state: RootState) => state.report);

  const [loading, setLoading] = useState<boolean>(true);
  const [circularProgressLoading, setCircularProgressLoading] = useState<boolean>(true);
  const [progressBarLoading, setProgressBarLoading] = useState<boolean>(true);
  const [response, setResponse] = useState<string | null>(null);

  let data;
  // let response;
  let riskScore;
  let questionList;
  let biometricDataList;

  if (route.params.type === 'reportStart') {
    // response = route.params.response;
    data = reportResult;
    riskScore = data.riskScore;
    questionList = data.questionList;
    biometricDataList = data.biometricDataList;
  } else {
    data = route.params.data;
    riskScore = data.risk_score;
    questionList = data.question_list;
    biometricDataList = data.biometric_data_list;

    // const createFeedback = async () => {
    //   const prompt = `
    //   다음은 ${name}님의 라이프스타일 및 생체정보 수치이며, 치매 위험도 예측에 활용됩니다. 각 항목에는 ${name}님의 수치와 치매환자 평균값이 함께 제공됩니다.

    //   - '작을수록 좋지 않은 항목'과 '클수록 좋지 않은 항목'이 있으며, 각각의 기준에 따라 평균값과의 차이가 큰 항목들을 분석해 주세요.
    //   - 평균값보다 **현저히 낮은 '작을수록 안 좋은 항목'**과, 평균값보다 **현저히 높은 '클수록 안 좋은 항목'** 중에서 **총 4개 항목을 선정**하세요.
    //   - 항목별로 개별적으로 나열하지 말고, **줄글 형태로 자연스럽게 이어지는 문장으로 분석 및 권장사항을 5~6문장 이내로 한국어로 작성**해 주세요.
    //   - 분석 내용은 명확하고 구체적이며, ${name}님이 자신의 상황을 이해하고 개선할 수 있도록 쉽게 설명해 주세요.
    //   - 수치는 아래 형식으로 제공됩니다: (${name}님 수치 / 치매환자 평균값)

    //   # 작을수록 안 좋은 항목:
    //   월 서적 지출, 하루간 활동 칼로리, 하루간 총 사용 칼로리, 총 활동 시간, 저강도 활동 시간, 중강도 활동 시간, 고강도 활동 시간, 하루간 평균 MET, 하루간 비활동 MET, 하루간 저강도 활동 MET, 하루간 중강도 활동 MET, 하루간 고강도 활동 MET, 하루간 걸음수, 하루간 움직인 거리, 총 수면 시간, 깊은 수면 시간, 렘 수면 시간, 수면 효율

    //   # 클수록 안 좋은 항목:
    //   월 음주 지출, 월 담배 지출, 비활동 시간, 수면 중 깬 시간, 얕은 수면 시간

    //   다음은 ${name}님 수치와 평균값입니다:

    //   1. 월 서적 지출: ${parseInt(questionList[12].answer, 10)} / 0.3
    //   2. 하루간 활동 칼로리: ${Math.floor((biometricDataList[1].biometric_data_value + biometricDataList[29].biometric_data_value + biometricDataList[57].biometric_data_value) / 3)} / 468
    //   3. 하루간 총 사용 칼로리: ${Math.floor((biometricDataList[2].biometric_data_value + biometricDataList[30].biometric_data_value + biometricDataList[58].biometric_data_value) / 3)} / 2519
    //   4. 총 활동 시간: ${Math.floor((biometricDataList[14].biometric_data_value + biometricDataList[42].biometric_data_value + biometricDataList[70].biometric_data_value) / 3)} / 341
    //   5. 저강도 활동 시간: ${Math.floor((biometricDataList[6].biometric_data_value + biometricDataList[34].biometric_data_value + biometricDataList[62].biometric_data_value) / 3)} / 285
    //   6. 중강도 활동 시간: ${Math.floor((biometricDataList[7].biometric_data_value + biometricDataList[35].biometric_data_value + biometricDataList[63].biometric_data_value) / 3)} / 53
    //   7. 고강도 활동 시간: ${Math.floor((biometricDataList[4].biometric_data_value + biometricDataList[32].biometric_data_value + biometricDataList[60].biometric_data_value) / 3)} / 3
    //   8. 하루간 평균 MET: ${Math.floor((biometricDataList[0].biometric_data_value + biometricDataList[28].biometric_data_value + biometricDataList[56].biometric_data_value) / 3)} / 1.5
    //   9. 하루간 비활동 MET: ${Math.floor((biometricDataList[9].biometric_data_value + biometricDataList[37].biometric_data_value + biometricDataList[65].biometric_data_value) / 3)} / 7
    //   10. 하루간 저강도 활동 MET: ${Math.floor((biometricDataList[10].biometric_data_value + biometricDataList[38].biometric_data_value + biometricDataList[66].biometric_data_value) / 3)} / 192
    //   11. 하루간 중강도 활동 MET: ${Math.floor((biometricDataList[11].biometric_data_value + biometricDataList[39].biometric_data_value + biometricDataList[67].biometric_data_value) / 3)} / 169
    //   12. 하루간 고강도 활동 MET: ${Math.floor((biometricDataList[8].biometric_data_value + biometricDataList[36].biometric_data_value + biometricDataList[64].biometric_data_value) / 3)} / 20
    //   13. 하루간 걸음수: ${Math.floor((biometricDataList[13].biometric_data_value + biometricDataList[41].biometric_data_value + biometricDataList[69].biometric_data_value) / 3)} / 10772
    //   14. 하루간 움직인 거리: ${Math.floor((biometricDataList[3].biometric_data_value + biometricDataList[31].biometric_data_value + biometricDataList[59].biometric_data_value) / 3)} / 8773
    //   15. 총 수면 시간: ${Math.floor((biometricDataList[27].biometric_data_value + biometricDataList[55].biometric_data_value + biometricDataList[83].biometric_data_value) / 3 / 60 / 60)} / 6
    //   16. 깊은 수면 시간: ${Math.floor((biometricDataList[16].biometric_data_value + biometricDataList[44].biometric_data_value + biometricDataList[72].biometric_data_value) / 3 / 60 / 60)} / 1
    //   17. 렘 수면 시간: ${Math.floor((biometricDataList[25].biometric_data_value + biometricDataList[53].biometric_data_value + biometricDataList[81].biometric_data_value) / 3 / 60 / 60)} / 1
    //   18. 수면 효율: ${Math.floor((biometricDataList[18].biometric_data_value + biometricDataList[46].biometric_data_value + biometricDataList[74].biometric_data_value) / 3)} / 81
    //   19. 월 음주 지출: ${parseInt(questionList[8].answer, 10)} / 0.7
    //   20. 월 담배 지출: ${parseInt(questionList[9].answer, 10)} / 1.5
    //   21. 비활동 시간: ${Math.floor((biometricDataList[5].biometric_data_value + biometricDataList[33].biometric_data_value + biometricDataList[61].biometric_data_value) / 3)} / 502
    //   22. 수면 중 깬 시간: ${Math.floor((biometricDataList[15].biometric_data_value + biometricDataList[43].biometric_data_value + biometricDataList[71].biometric_data_value) / 3 / 60 / 60)} / 1
    //   23. 얕은 수면 시간: ${Math.floor((biometricDataList[21].biometric_data_value + biometricDataList[49].biometric_data_value + biometricDataList[77].biometric_data_value) / 3 / 60 / 60)} / 4
    //   `;

    //   const result = await sendMessageToChatGPT(prompt);
    //   setResponse(result);
    // };

    // createFeedback();
  }

  useEffect(() => {
    if (route.params.type !== 'reportStart') {
      const createFeedback = async () => {
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

        1. 월 서적 지출: ${parseInt(questionList[12].answer, 10)} / 0.3
        2. 하루간 활동 칼로리: ${Math.floor((biometricDataList[1].biometric_data_value + biometricDataList[29].biometric_data_value + biometricDataList[57].biometric_data_value) / 3)} / 468
        3. 하루간 총 사용 칼로리: ${Math.floor((biometricDataList[2].biometric_data_value + biometricDataList[30].biometric_data_value + biometricDataList[58].biometric_data_value) / 3)} / 2519
        4. 총 활동 시간: ${Math.floor((biometricDataList[14].biometric_data_value + biometricDataList[42].biometric_data_value + biometricDataList[70].biometric_data_value) / 3)} / 341
        5. 저강도 활동 시간: ${Math.floor((biometricDataList[6].biometric_data_value + biometricDataList[34].biometric_data_value + biometricDataList[62].biometric_data_value) / 3)} / 285
        6. 중강도 활동 시간: ${Math.floor((biometricDataList[7].biometric_data_value + biometricDataList[35].biometric_data_value + biometricDataList[63].biometric_data_value) / 3)} / 53
        7. 고강도 활동 시간: ${Math.floor((biometricDataList[4].biometric_data_value + biometricDataList[32].biometric_data_value + biometricDataList[60].biometric_data_value) / 3)} / 3
        8. 하루간 평균 MET: ${Math.floor((biometricDataList[0].biometric_data_value + biometricDataList[28].biometric_data_value + biometricDataList[56].biometric_data_value) / 3)} / 1.5
        9. 하루간 비활동 MET: ${Math.floor((biometricDataList[9].biometric_data_value + biometricDataList[37].biometric_data_value + biometricDataList[65].biometric_data_value) / 3)} / 7
        10. 하루간 저강도 활동 MET: ${Math.floor((biometricDataList[10].biometric_data_value + biometricDataList[38].biometric_data_value + biometricDataList[66].biometric_data_value) / 3)} / 192
        11. 하루간 중강도 활동 MET: ${Math.floor((biometricDataList[11].biometric_data_value + biometricDataList[39].biometric_data_value + biometricDataList[67].biometric_data_value) / 3)} / 169
        12. 하루간 고강도 활동 MET: ${Math.floor((biometricDataList[8].biometric_data_value + biometricDataList[36].biometric_data_value + biometricDataList[64].biometric_data_value) / 3)} / 20
        13. 하루간 걸음수: ${Math.floor((biometricDataList[13].biometric_data_value + biometricDataList[41].biometric_data_value + biometricDataList[69].biometric_data_value) / 3)} / 10772
        14. 하루간 움직인 거리: ${Math.floor((biometricDataList[3].biometric_data_value + biometricDataList[31].biometric_data_value + biometricDataList[59].biometric_data_value) / 3)} / 8773
        15. 총 수면 시간: ${Math.floor((biometricDataList[27].biometric_data_value + biometricDataList[55].biometric_data_value + biometricDataList[83].biometric_data_value) / 3 / 60 / 60)} / 6
        16. 깊은 수면 시간: ${Math.floor((biometricDataList[16].biometric_data_value + biometricDataList[44].biometric_data_value + biometricDataList[72].biometric_data_value) / 3 / 60 / 60)} / 1
        17. 렘 수면 시간: ${Math.floor((biometricDataList[25].biometric_data_value + biometricDataList[53].biometric_data_value + biometricDataList[81].biometric_data_value) / 3 / 60 / 60)} / 1
        18. 수면 효율: ${Math.floor((biometricDataList[18].biometric_data_value + biometricDataList[46].biometric_data_value + biometricDataList[74].biometric_data_value) / 3)} / 81
        19. 월 음주 지출: ${parseInt(questionList[8].answer, 10)} / 0.7
        20. 월 담배 지출: ${parseInt(questionList[9].answer, 10)} / 1.5
        21. 비활동 시간: ${Math.floor((biometricDataList[5].biometric_data_value + biometricDataList[33].biometric_data_value + biometricDataList[61].biometric_data_value) / 3)} / 502
        22. 수면 중 깬 시간: ${Math.floor((biometricDataList[15].biometric_data_value + biometricDataList[43].biometric_data_value + biometricDataList[71].biometric_data_value) / 3 / 60 / 60)} / 1
        23. 얕은 수면 시간: ${Math.floor((biometricDataList[21].biometric_data_value + biometricDataList[49].biometric_data_value + biometricDataList[77].biometric_data_value) / 3 / 60 / 60)} / 4
        `;
        const result = await sendMessageToChatGPT(prompt);
        setResponse(result);
        setLoading(false);
      };

      createFeedback();
      // console.log(response);
    } else {
      setResponse(route.params.response);
      setLoading(false);
    }
  }, []);

//   useEffect(() => {
//   if (progressValue === 1) {
//     setLoading(false);
//   }
// }, [progressValue]);

  return (
    <View style={styles.container}>
      {loading && circularProgressLoading && progressBarLoading ? (
        <View style={{ flex: 1, alignItems: 'center',  justifyContent: 'center' }}>
          <ActivityIndicator size="large" color="#888" />
        </View>
      ) : (

        <>
      <SafeAreaView style={styles.safeAreaWrapper}>
        <TouchableOpacity onPress={() => {route.params.type === 'reportView' ? navigation.replace('ReportView', { from: 'ReportResult' }) : navigation.replace('Home');}}>
          <Icon name="chevron-back" size={16} color="gray" />
        </TouchableOpacity>
      </SafeAreaView>

      <ScrollView showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.box}>
          <CustomText style={styles.resultBoxText}>
            {name}님의 치매 날씨,{'\n'}
            <CustomText style={styles.resultText}>
              {riskScore === null ? ''
              : riskScore >= 80 ? '☁️ 매우 흐림 '
              : riskScore >= 60 ? '⛅ 흐림 '
              : riskScore >= 50 ? '🌤️ 약간 흐림 '
              : '☀️ 맑음 '
              }
            </CustomText>
            이에요.
          </CustomText>
        </View>

        <View style={styles.section}>
          <CustomText style={styles.sectionTitle}>치매 위험도</CustomText>
          <View style={styles.circleContainer}>
            <AnimatedCircularProgress
              size={160}
              width={40}
              fill={riskScore || 0}
              // tintColor="#D6CCC2"
              tintColor={riskScore === null ? ''
              : riskScore >= 80 ? '#B76C6C'
              : riskScore >= 60 ? '#C59292'
              : riskScore >= 50 ? '#EFE09F'
              : '#98A798'
              }
              backgroundColor="#F2EFED"
              rotation={0}
              lineCap="butt"
              onAnimationComplete={() => {
                // ✅ 애니메이션 완료
                // setCircularProgressLoading(false);
                setTimeout(() => {
                  setCircularProgressLoading(false);
                }, 0);
              }}
            >
              {(fill: number) => <CustomText style={styles.percentText}>{Math.round(fill)}%</CustomText>}
            </AnimatedCircularProgress>
          </View>
        </View>

        <View style={styles.section}>
          <CustomText style={styles.sectionTitle}>라이프스타일</CustomText>
          <View style={styles.sectionContent}>
            {renderBar('가구원 수', parseInt(questionList[4].answer, 10), 10, 2)}
            {renderBar('월 소득', parseInt(questionList[6].answer, 10), 500, 243)}
            {renderBar('월 지출', parseInt(questionList[7].answer, 10), 500, 207)}
            {renderBar('월 음주 지출', parseInt(questionList[8].answer, 10), 20, 0.7)}
            {renderBar('월 담배 지출', parseInt(questionList[9].answer, 10), 20, 1.5)}
            {renderBar('월 서적 지출', parseInt(questionList[12].answer, 10), 20, 0.3)}
            {renderBar('월 의료 지출', parseInt(questionList[10].answer, 10), 50, 21)}
            {renderBar('월 보험 지출', parseInt(questionList[14].answer, 10), 50, 5)}
          </View>
        </View>

        <View style={styles.section}>
          <CustomText style={styles.sectionTitle}>생체정보</CustomText>
          <View style={styles.sectionContent}>
            <View style={styles.sectionContentTitleWrapper}>
              <Image source={require('../../assets/images/fire.png')} style={styles.image} />
              <CustomText style={styles.sectionContentTitle}>칼로리</CustomText>
            </View>
            {renderBar('하루간 활동 칼로리 (kcal)', Math.floor((biometricDataList[1].biometric_data_value + biometricDataList[29].biometric_data_value + biometricDataList[57].biometric_data_value) / 3), 2000, 468)}
            {renderBar('하루간 총 사용 칼로리 (kcal)', Math.floor((biometricDataList[2].biometric_data_value + biometricDataList[30].biometric_data_value + biometricDataList[58].biometric_data_value) / 3), 4500, 2519)}
          </View>

          <View style={styles.sectionContent}>
            <View style={styles.sectionContentTitleWrapper}>
              <Image source={require('../../assets/images/clock.png')} style={styles.image} />
              <CustomText style={styles.sectionContentTitle}>활동 시간</CustomText>
            </View>
            {renderBar('총 활동 시간 (분)', Math.floor((biometricDataList[14].biometric_data_value + biometricDataList[42].biometric_data_value + biometricDataList[70].biometric_data_value) / 3), 1000, 341)}
            {renderBar('비활동 시간 (분)', Math.floor((biometricDataList[5].biometric_data_value + biometricDataList[33].biometric_data_value + biometricDataList[61].biometric_data_value) / 3), 1000, 502)}
            {renderBar('저강도 활동 시간 (분)', Math.floor((biometricDataList[6].biometric_data_value + biometricDataList[34].biometric_data_value + biometricDataList[62].biometric_data_value) / 3), 1000, 285)}
            {renderBar('중강도 활동 시간 (분)', Math.floor((biometricDataList[7].biometric_data_value + biometricDataList[35].biometric_data_value + biometricDataList[63].biometric_data_value) / 3), 200, 53)}
            {renderBar('고강도 활동 시간 (분)', Math.floor((biometricDataList[4].biometric_data_value + biometricDataList[32].biometric_data_value + biometricDataList[60].biometric_data_value) / 3), 60, 3)}
            {renderBar('미착용 시간 (분)', Math.floor((biometricDataList[12].biometric_data_value + biometricDataList[40].biometric_data_value + biometricDataList[68].biometric_data_value) / 3), 200, 44)}
          </View>

          <View style={styles.sectionContent}>
            <View style={styles.sectionContentTitleWrapper}>
              <Image source={require('../../assets/images/strength.png')} style={styles.image} />
              <CustomText style={styles.sectionContentTitle}>활동 MET</CustomText>
            </View>
            {renderBar('하루간 평균 MET', Math.floor((biometricDataList[0].biometric_data_value + biometricDataList[28].biometric_data_value + biometricDataList[56].biometric_data_value) / 3), 20, 1.5)}
            {renderBar('하루간 비활동 MET', Math.floor((biometricDataList[9].biometric_data_value + biometricDataList[37].biometric_data_value + biometricDataList[65].biometric_data_value) / 3), 20, 7)}
            {renderBar('하루간 저강도 활동 MET', Math.floor((biometricDataList[10].biometric_data_value + biometricDataList[38].biometric_data_value + biometricDataList[66].biometric_data_value) / 3), 300, 192)}
            {renderBar('하루간 중강도 활동 MET', Math.floor((biometricDataList[11].biometric_data_value + biometricDataList[39].biometric_data_value + biometricDataList[67].biometric_data_value) / 3), 300, 169)}
            {renderBar('하루간 고강도 활동 MET', Math.floor((biometricDataList[8].biometric_data_value + biometricDataList[36].biometric_data_value + biometricDataList[64].biometric_data_value) / 3), 300, 20)}
          </View>

          <View style={styles.sectionContent}>
            <View style={styles.sectionContentTitleWrapper}>
              <Image source={require('../../assets/images/walking.png')} style={styles.image} />
              <CustomText style={styles.sectionContentTitle}>걷기</CustomText>
            </View>
            {renderBar('하루간 걸음수', Math.floor((biometricDataList[13].biometric_data_value + biometricDataList[41].biometric_data_value + biometricDataList[69].biometric_data_value) / 3), 50000, 10772)}
            {renderBar('하루간 움직인 거리 (m)', Math.floor((biometricDataList[3].biometric_data_value + biometricDataList[31].biometric_data_value + biometricDataList[59].biometric_data_value) / 3), 20000, 8773)}
          </View>

          <View style={styles.sectionContent}>
            <View style={styles.sectionContentTitleWrapper}>
              <Image source={require('../../assets/images/snooze.png')} style={styles.image} />
              <CustomText style={styles.sectionContentTitle}>수면</CustomText>
            </View>
            {renderBar('수면 중 깬 시간 (시간)', Math.floor((biometricDataList[15].biometric_data_value + biometricDataList[43].biometric_data_value + biometricDataList[71].biometric_data_value) / 3 / 60 / 60), 12, Math.floor(5977 / 60 / 60))}
            {renderBar('얕은 수면 시간 (시간)', Math.floor((biometricDataList[21].biometric_data_value + biometricDataList[49].biometric_data_value + biometricDataList[77].biometric_data_value) / 3 / 60 / 60), 12, Math.floor(15444 / 60 / 60))}
            {renderBar('깊은 수면 시간 (시간)', Math.floor((biometricDataList[16].biometric_data_value + biometricDataList[44].biometric_data_value + biometricDataList[72].biometric_data_value) / 3 / 60 / 60), 12, Math.floor(4907 / 60 / 60))}
            {renderBar('렘 수면 시간 (시간)', Math.floor((biometricDataList[25].biometric_data_value + biometricDataList[53].biometric_data_value + biometricDataList[81].biometric_data_value) / 3 / 60 / 60), 12, Math.floor(3617 / 60 / 60))}
            {renderBar('총 수면 시간 (시간)', Math.floor((biometricDataList[27].biometric_data_value + biometricDataList[55].biometric_data_value + biometricDataList[83].biometric_data_value) / 3 / 60 / 60), 12, Math.floor(23970 / 60 / 60))}

            {/* {renderBar('수면 중 깬 시간', (biometricDataList[15].biometric_data_value + biometricDataList[43].biometric_data_value + biometricDataList[71].biometric_data_value) / 3, 12, 5977)}
            {renderBar('얕은 수면 시간', (biometricDataList[21].biometric_data_value + biometricDataList[49].biometric_data_value + biometricDataList[77].biometric_data_value) / 3, 12, 15444)}
            {renderBar('깊은 수면 시간', (biometricDataList[16].biometric_data_value + biometricDataList[44].biometric_data_value + biometricDataList[72].biometric_data_value) / 3, 12, 4907)}
            {renderBar('렘 수면 시간', (biometricDataList[25].biometric_data_value + biometricDataList[53].biometric_data_value + biometricDataList[81].biometric_data_value) / 3, 12, 3617)}
            {renderBar('총 수면 시간', (biometricDataList[27].biometric_data_value + biometricDataList[55].biometric_data_value + biometricDataList[83].biometric_data_value) / 3, 12, 23970)} */}

            {/* {renderBar('수면 중간점 시간', Math.floor((biometricDataList[23].biometric_data_value + biometricDataList[51].biometric_data_value + biometricDataList[79].biometric_data_value) / 3 / 60 / 60), 30000, Math.floor(15022 / 60 / 60))} */}
            {renderBar('수면 중간점 시간 델타', Math.floor((biometricDataList[22].biometric_data_value + biometricDataList[50].biometric_data_value + biometricDataList[78].biometric_data_value) / 3 / 60 / 60), 12, Math.floor(11206 / 60 / 60))}
            {renderBar('수면 효율', Math.floor((biometricDataList[18].biometric_data_value + biometricDataList[46].biometric_data_value + biometricDataList[74].biometric_data_value) / 3), 100, 81)}
            {renderBar('분당 평균 수면 심박수', Math.floor((biometricDataList[19].biometric_data_value + biometricDataList[47].biometric_data_value + biometricDataList[75].biometric_data_value) / 3), 120, 60)}
            {renderBar('분당 평균 수면 심박변동', Math.floor((biometricDataList[26].biometric_data_value + biometricDataList[54].biometric_data_value + biometricDataList[82].biometric_data_value) / 3), 120, 32)}
            </View>
        </View>

        <View style={styles.section}>
          <CustomText style={styles.sectionTitle}>개선사항</CustomText>
          {/* {loading ? (
            <View style={styles.box}>
              <ActivityIndicator size="small" color="#888" />
            </View>
          ) : ( */}
            <View style={styles.box}>
              <CustomText style={styles.feedbackBoxText}>
              {response?.split('\n').map((paragraph, index, array) => (
                <CustomText key={index} style={styles.content}>
                  {'  '}{paragraph}{index !== array.length - 1 ? '\n' : ''}
                </CustomText>
              ))}
              </CustomText>
            </View>
          {/* )} */}
        </View>
      </ScrollView>
      </>
      )}
    </View>
  );
};

const renderBar = (label: string, value: number, maxValue: number, averageValue: number) => {
  const progress = value / maxValue;
  const currentValue = Math.round(value);
  // let averagePosition = 0;
  // if (label.includes('수면') && label.includes('시간') && !label.includes('중간점')) {
  //   averagePosition = (averageValue / 60 / 60 / maxValue) * 100;
  // } else {
  //   averagePosition = (averageValue / maxValue) * 100;
  // }
  const averagePosition = (averageValue / maxValue) * 100;

  return (
    <View style={styles.barWrapper} key={label}>
      <CustomText style={styles.barLabel}>{label}</CustomText>

      <View style={styles.barWithMarker}>
        <View style={[styles.averageMarker, { left: `${averagePosition}%` }]} />
        <CustomText style={[styles.averageText, { left: `${averagePosition}%` }]}>{averageValue}</CustomText>
        {/* {label.includes('수면') && label.includes('시간') && !label.includes('중간점') ?
        <CustomText style={[styles.averageText, { left: `${averagePosition}%` }]}>{Math.floor(averageValue / 60 / 60)}시간 {averageValue % 60}분</CustomText>
        :
        <CustomText style={[styles.averageText, { left: `${averagePosition}%` }]}>{averageValue}</CustomText>
        } */}
        <ProgressBar progress={progress} color="#D6CCC2" style={styles.bar} />
        {currentValue !== 0 && <CustomText style={[styles.barCurrentValue, { left: `${progress * 100 + 1}%` }]}>{currentValue}</CustomText>}
        {/* {label.includes('수면') && label.includes('시간') && !label.includes('중간점') ?
        <CustomText style={[styles.barCurrentValue, { left: `${progress * 100 + 1}%` }]}>{Math.floor(currentValue / 60 / 60)}시간 {currentValue % 60}분</CustomText>
        :
        <CustomText style={[styles.barCurrentValue, { left: `${progress * 100 + 1}%` }]}>{currentValue}</CustomText>
        } */}
      </View>

      <View style={styles.barNumbers}>
        <CustomText style={[styles.barNumber, { transform: [{ translateX: -4 }] }]}>0</CustomText>
        {/* {currentValue !== 0 && <CustomText style={[styles.barCurrentValue, { left: `${progress * 100}%` }]}>{currentValue}</CustomText>} */}
        <CustomText style={[styles.barNumber, { transform: [{ translateX: 4 }] }]}>{maxValue}</CustomText>
      </View>
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
  safeAreaWrapper: {
    width: 70,
    paddingVertical: 16,
    paddingHorizontal: 10,
  },
  box: {
    alignItems: 'center',
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  resultBoxText: {
    textAlign: 'center',
    fontSize: 22,
    lineHeight: 35,
  },
  resultText: {
    fontSize: 22,
    color: '#917A6B',
    lineHeight: 35,
  },
  section: {
    marginTop: 20,
  },
  sectionTitle: {
    fontSize: 24,
    marginBottom: 16,
  },
  circleContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  percentText: {
    fontSize: 30,
  },
  sectionContent: {
    padding: 10,
    marginBottom: 15,
    // backgroundColor: '#F2EAE3',
    backgroundColor: '#F2EFED',
    borderRadius: 10,
  },
  sectionContentTitleWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  sectionContentTitle: {
    fontSize: 22,
    marginBottom: 5,
  },
  image: {
    width: 18,
    height: 18,
    marginRight: 5,
  },
  feedbackBoxText: {
    fontSize: 18,
  },
  barWrapper: {
    marginBottom: 16,
  },
  barLabel: {
    fontSize: 18,
    marginBottom: 12,
  },
  bar: {
    height: 20,
    borderRadius: 7,
    backgroundColor: '#E2E2E2',
    zIndex: 1,
  },
  barWithMarker: {
    position: 'relative',
    justifyContent: 'center',
  },
  averageMarker: {
    position: 'absolute',
    top: 0,
    width: 4,
    height: 20,
    backgroundColor: '#B94141',
    zIndex: 2,
  },
  averageText: {
    position: 'absolute',
    top: 24,
    fontSize: 12,
    color: '#B94141',
    // transform: [{ translateX: -1 }],
  },
  barNumbers: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
    marginTop: 4,
  },
  barNumber: {
    fontSize: 12,
    color: 'gray',
  },
  barCurrentValue: {
    position: 'absolute',
    // left: 20,
    fontSize: 12,
    color: 'black',
    textAlign: 'center',
    zIndex: 3,
    // transform: [{ translateX: -8 }],
  },
  content: {
    lineHeight: 25,
    color: '#434240',
  },
});

export default ReportResultScreen;