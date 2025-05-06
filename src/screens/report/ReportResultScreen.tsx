import React from 'react';
import { View, SafeAreaView, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { AnimatedCircularProgress } from 'react-native-circular-progress';
import { ProgressBar } from 'react-native-paper';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const ReportResultScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'ReportResult'>;
  const navigation = useNavigation<Navigation>();

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeAreaWrapper}>
        <TouchableOpacity onPress={() => {navigation.replace('Home')}}>
          <Icon name="chevron-back" size={16} />
        </TouchableOpacity>
      </SafeAreaView>

      <ScrollView showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.box}>
          <CustomText style={styles.resultBoxText}>
            송민하님의 치매 날씨,{'\n'}
            <CustomText style={styles.resultText}>⛅약간 흐림</CustomText>
            이에요.
          </CustomText>
        </View>

        <View style={styles.section}>
          <CustomText style={styles.sectionTitle}>치매 위험도</CustomText>
          <View style={styles.circleContainer}>
            <AnimatedCircularProgress
              size={160}
              width={40}
              fill={60}
              tintColor="#D6CCC2"
              backgroundColor="#F2EFED"
              rotation={0}
              lineCap="butt"
            >
              {(fill: number) => <CustomText style={styles.percentText}>{Math.round(fill)}%</CustomText>}
            </AnimatedCircularProgress>
          </View>
        </View>

        <View style={styles.section}>
          <CustomText style={styles.sectionTitle}>라이프스타일</CustomText>
          <View style={styles.sectionContent}>
            {renderBar('가구원 수')}
            {renderBar('월 소득')}
            {renderBar('월 지출')}
            {renderBar('월 음주 지출')}
            {renderBar('월 담배 지출')}
            {renderBar('월 서적 지출')}
            {renderBar('월 의료 지출')}
            {renderBar('월 보험 지출')}
          </View>
        </View>

        <View style={styles.section}>
          <CustomText style={styles.sectionTitle}>생체정보</CustomText>
          <View style={styles.sectionContent}>
            {renderBar('하루간 활동 칼로리')}
            {renderBar('하루간 총 사용 칼로리')}
            {renderBar('매일 움직인 거리')}
            {renderBar('활동 종료 시간')}
            {renderBar('활동 시작 시간')}
            {renderBar('비활동 시간')}
            {renderBar('저강도 활동 시간')}
            {renderBar('중강도 활동 시간')}
            {renderBar('고강도 활동 시간')}
            {renderBar('하루간 비활동 MET')}
            {renderBar('하루간 저강도 활동 MET')}
            {renderBar('하루간 중강도 활동 MET')}
            {renderBar('하루간 고강도 활동 MET')}
            {renderBar('하루간 1분당 MET 로그')}
            {renderBar('미착용 시간')}
          </View>
        </View>

        <View style={styles.section}>
          <CustomText style={styles.sectionTitle}>개선사항</CustomText>
          <View style={styles.box}>
            <CustomText style={styles.feedbackBoxText}>
              월 음주 지출이 치매 환자들의 평균보다 높고 수면 시간이 치매 환자들의 평균보다 낮아요. 음주를 줄이고 수면 시간을 늘리는 등 개선이 필요해보여요!
            </CustomText>
          </View>
        </View>
      </ScrollView>
    </View>
  );
};

const renderBar = (label: string) => {
  return (
    <View style={styles.barWrapper} key={label}>
      <CustomText style={styles.barLabel}>{label}</CustomText>
      <ProgressBar progress={0.6} color="#D6CCC2" style={styles.bar} />
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
    marginLeft: 5,
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
    backgroundColor: '#F2EFED',
  },
});

export default ReportResultScreen;