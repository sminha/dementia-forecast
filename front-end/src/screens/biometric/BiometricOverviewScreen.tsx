import React, { useRef, useState, useEffect } from 'react';
import { ScrollView, TouchableOpacity, Animated, Pressable, Image, StyleSheet, View, Dimensions } from 'react-native';
import { useSelector } from 'react-redux';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { RootState } from '../../redux/store.ts';
import { loadTokens, logAllAsyncStorage } from '../../redux/actions/authAction.ts';
import { LineChart, BarChart } from 'react-native-chart-kit';
import CustomText from '../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';
// import { read } from 'react-native-fs';

const SCREEN_WIDTH = Dimensions.get('window').width;
const TAB_WIDTH = (SCREEN_WIDTH - 32) / 3;

// 날짜
const formatToTab = (date: string): string => {
  const [year, month, day] = date.split('-');
  const formatted = `${parseInt(month, 10)}월 ${parseInt(day, 10)}일`;

  return formatted;
};

// 걷기
const intervalMinutes = 10;
const intervalsPerDay = (24 * 60) / intervalMinutes;

const labels = Array.from({ length: intervalsPerDay }, (_, i) => {
  const totalMinutes = i * intervalMinutes;
  const hour = String(Math.floor(totalMinutes / 60)).padStart(2, '0');
  const min = String(totalMinutes % 60).padStart(2, '0');
  return `${hour}:${min}`;
});

const labelInterval = 6;
const displayedLabels = labels.map((label, i) => (i % labelInterval === 0 ? label : ''));

// 수면
const totalMinutesforSleep = 600;

const timeToMinutes = (time: string): number => {
  const [h, m] = time.split(':').map(Number);
  const minutes = h * 60 + m;
  return h < 12 ? minutes + 1440 : minutes;
};

const baseStart = 22 * 60;

function formatToHourMinute(dateString: string) {
  const date = new Date(dateString.replace(' ', 'T'));
  const hour = String(date.getHours()).padStart(2, '0');
  const minute = String(date.getMinutes()).padStart(2, '0');
  return `${hour}:${minute}`;
}

const stageMap = {
  40001: { label: '수면 중 깸', color: '#B94141' },
  40002: { label: '얕은 수면', color: '#D6CCC2' },
  40003: { label: '깊은 수면', color: '#B79E8E' },
  40004: { label: '렘 수면', color: '#917A6B' },
};

const convertSleepLogsToStages = (sleepLogs) => {
  return sleepLogs.map(log => {
    const start = log.start_time.slice(11,16);
    const end = log.end_time.slice(11,16);
    const stageInfo = stageMap[log.stage] || { label: '알수없음', color: '#999' };
    return {
      label: stageInfo.label,
      start,
      end,
      color: stageInfo.color,
    };
  });
};

// const readBiometricInfo = async () => {
//   const biometricInfoInStorage = await AsyncStorage.getItem('biometricInfo');
//   if (!biometricInfoInStorage) {
//     console.log('생체 정보가 없습니다.');
//     return;
//   }

//   const biometricInfo = JSON.parse(biometricInfoInStorage);

//   return biometricInfo;
// };

const BiometricOverviewScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'BiometricOverview'>;
  const navigation = useNavigation<Navigation>();

  const route = useRoute<RouteProp<RootStackParamList, 'BiometricOverview'>>();
  const from = route.params?.from;

  // logAllAsyncStorage();
  // useEffect(() => {
  //   const fetchData = async () => {
  //     const result = await readBiometricInfo();
  //     console.log('결과', result);
  //   };

  //   console.log('얿음');
  //   fetchData();
  // }, []);

  const biometric = useSelector((state: RootState) => state.biometric);
  const biometricInfo = useSelector((state: RootState) => state.biometric.biometricData);
  // const biometricInfo = biometricInfoAll.biometricData;

  const userInfo = useSelector((state: RootState) => state.user.userInfo);

  const [selectedIndex, setSelectedIndex] = useState(0);

  const translateX = useRef(new Animated.Value(TAB_WIDTH * selectedIndex)).current;

  // 날짜
  // const TABS = [{ date: formatToTab(biometricInfo[0].date) }, { date: formatToTab(biometricInfo[1].date) }, { date: '5월 21일' }]; // FOR TEST - 연속 2일치
  const TABS = [{ date: formatToTab(biometricInfo[0].date) }, { date: formatToTab(biometricInfo[1]?.date) }, { date: formatToTab(biometricInfo[2]?.date) }];

  // 레이블
  const start = new Date('2000-01-01T00:00:00');
  const end = new Date('2000-01-01T23:59:59');
  const fullTimeLabels: string[] = [];
  for (let time = new Date(start); time <= end; time.setMinutes(time.getMinutes() + 10)) {
    const hour = time.getHours().toString().padStart(2, '0');
    const minute = time.getMinutes().toString().padStart(2, '0');
    fullTimeLabels.push(`${hour}:${minute}`);
  } // 00:00, 00:10, 00:20, 00:30, ...
  const displayLabels = fullTimeLabels.map((label, i) => (i % 6 === 0 ? label : '')); // 00:00, 01:00, 02:00, 03:00, ...

  // biometricInfo[0], [1], [2]- 1, 2, 3일치
  // .biometric_data_list[0], [1], [2], [3]- exercise, walk, sleep, heartRate

  // 심박수
  const averageHeartRate = (biometricInfo[selectedIndex].biometric_data_list[3].biometric_data_value.summary.average_heart_rate).toFixed(0);
  const heartRateLogs = biometricInfo[selectedIndex].biometric_data_list[3].biometric_data_value.heart_rate_logs;

  const timeLabels = heartRateLogs.map(log => {
    const date = new Date(log.start_time);
    const hour = String(date.getHours()).padStart(2, '0');
    const minute = String(date.getMinutes()).padStart(2, '0');
    return `${hour}:${minute}`;
  });

  const heartRateMap = new Map();
  heartRateLogs.forEach(log => {
    const isoString = log.start_time.replace(' ', 'T');
    const time = new Date(isoString);
    const hour = time.getHours().toString().padStart(2, '0');
    const minute = time.getMinutes().toString().padStart(2, '0');
    const key = `${hour}:${minute}`;
    heartRateMap.set(key, log.heart_rate);
  });

  const alignedHeartRate = fullTimeLabels.map(label => { return heartRateMap.get(label) ?? 0; });

  // 활동
  const totalExerciseDuration = biometricInfo[selectedIndex].biometric_data_list[0].biometric_data_value.reduce((acc, cur) => acc + cur.duration, 0) / 1000 / 60;
  const totalExerciseDurationHr = Math.floor(totalExerciseDuration / 60);
  const totalExerciseDurationMin = (totalExerciseDuration).toFixed(0);
  const totalCalorie = biometricInfo[selectedIndex].biometric_data_list[0].biometric_data_value.reduce((acc, cur) => acc + cur.calorie, 0);
  const exerciseLogs = biometricInfo[selectedIndex].biometric_data_list[0].biometric_data_value;

  // 걷기
  const totalStepCount = biometricInfo[selectedIndex].biometric_data_list[1].biometric_data_value.summary.total_step_count;
  const totalDistance = biometricInfo[selectedIndex].biometric_data_list[1].biometric_data_value.summary.total_distance;
  const totalDistanceKm = Math.floor(totalDistance / 1000);
  const totalDistanceM = (totalDistance % 1000).toFixed(0);

  const stepCounts = Array.from({ length: intervalsPerDay }, () => 0);
  const distanceData = Array.from({ length: intervalsPerDay }, () => 0);

  const stepLogs = biometricInfo[selectedIndex].biometric_data_list[1].biometric_data_value?.step_logs ?? [];

  stepLogs.forEach(({ start_time, step_count }) => {
    const date = new Date(start_time);
    const minutesSinceMidnight = date.getHours() * 60 + date.getMinutes();
    const intervalIndex = Math.floor(minutesSinceMidnight / intervalMinutes);
    if (intervalIndex >= 0 && intervalIndex < intervalsPerDay) {
      stepCounts[intervalIndex] += step_count;
    }
  });

  stepLogs.forEach(({ start_time, distance }) => {
    const date = new Date(start_time);
    const minutesSinceMidnight = date.getHours() * 60 + date.getMinutes();
    const intervalIndex = Math.floor(minutesSinceMidnight / intervalMinutes);
    if (intervalIndex >= 0 && intervalIndex < intervalsPerDay) {
      distanceData[intervalIndex] += distance;
    }
  });

  const roundedData = distanceData.map((value) => Math.floor(value));

  // 수면
  const totalSleepDuration = biometricInfo[selectedIndex].biometric_data_list[2].biometric_data_value.summary.total_duration;
  const totalSleepDurationHour = Math.floor(totalSleepDuration / 60);
  const totalSleepDurationMin = (totalSleepDuration % 60).toFixed(0);
  const efficiency = biometricInfo[selectedIndex].biometric_data_list[2].biometric_data_value.summary.efficiency;

  const sleepStages = convertSleepLogsToStages(biometricInfo[selectedIndex].biometric_data_list[2].biometric_data_value.sleep_logs);

  const preferredOrder = ['수면 중 깸', '얕은 수면', '깊은 수면', '렘 수면'];

  const aggregatedStages = sleepStages.reduce((acc, stage) => {
    const duration = timeToMinutes(stage.end) - timeToMinutes(stage.start);
    if (!acc[stage.label]) {
      acc[stage.label] = { label: stage.label, color: stage.color, duration: 0 };
    }
    acc[stage.label].duration += duration;
    return acc;
  }, {});

  const aggregatedStageList = preferredOrder
  .filter(label => aggregatedStages[label])
  .map(label => aggregatedStages[label]);

  const totalDuration = aggregatedStageList.reduce((sum, stage) => sum + stage.duration, 0);

  const onTabPress = (index: number) => {
    setSelectedIndex(index);
    Animated.timing(translateX, {
      toValue: TAB_WIDTH * index,
      duration: 200,
      useNativeDriver: true,
    }).start();
  };

  useEffect(() => {
    const saveBiometricDataToStorage = async () => {
      try {
        // await AsyncStorage.setItem('biometricInfo', JSON.stringify({ email: userInfo.email, biometricInfo: biometricInfo}));
        const now = Date.now();
        const dataToStore = {
          email: userInfo.email,
          biometricInfo: biometricInfo,
          // biometricInfo: biometric,
          savedAt: now,
        };
        await AsyncStorage.setItem('biometricInfo', JSON.stringify(dataToStore));
      } catch (error) {
        console.error('생체정보 저장 실패:', error);
      }
    };

    saveBiometricDataToStorage();
  }, []);

  // useEffect(() => {
  //   const handlePredict = async () => {
  //     const biometricInfoInStorage = await AsyncStorage.getItem('biometricInfo');
  //     if (!biometricInfoInStorage) {
  //       console.log('생체 정보가 없습니다.');
  //       return;
  //     }

  //     const biometricInfo = JSON.parse(biometricInfoInStorage);

  //     return biometricInfo;
  //   };

  //   handlePredict();
  // });

  // const handlePredict = async () => {
  //   const { accessToken } = await loadTokens();
  //   if (!accessToken) {
  //     console.log('로그인 정보가 없습니다.');
  //     return;
  //   }

  //   const biometricInfoInStorage = await AsyncStorage.getItem('biometricInfo');
  //   if (!biometricInfoInStorage) {
  //     console.log('생체 정보가 없습니다.');
  //     return;
  //   }

  //   const biometricInfo = JSON.parse(biometricInfoInStorage);

  //   const resultAction = await dispatch(createReport({ token: accessToken, biometricInfo }));

  //   if (createReport.fulfilled.match(resultAction)) {
  //     const { message } = resultAction.payload;
  //     console.log('✅', message);
  //     // 성공 처리 렌더링 로직
  //   } else if (createReport.rejected.match(resultAction)) {
  //     const errorMessage =
  //       (resultAction.payload as any)?.message || '레포트 생성 실패';
  //     console.log('❌', errorMessage);
  //     // 실패 처리 렌더링 로직
  //   }
  // };

    // console.log('신장, 체중 입력 후 biometric:', biometric);
    // console.log('신장, 체중 입력 후 biometricData:', biometricInfo);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          <TouchableOpacity
            style={styles.backContainer}
            onPress={() => {
              if (from === 'Mypage') {
                // navigation.replace('Mypage');
                navigation.goBack();
              } else {
                navigation.replace('Home');
              }
            }}
          >
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity>
        </View>

        <View style={styles.sliderContainer}>
          <Animated.View style={[styles.slider, { transform: [{ translateX }]}]} />
          {TABS.map((tab, index) => {
            const isSelected = index === selectedIndex;
            return (
              <Pressable
                key={index}
                onPress={() => onTabPress(index)}
                style={styles.tab}
              >
                <CustomText style={[styles.date, { color: isSelected ? '#434240' : '#B4B4B4' }]}>{tab.date}</CustomText>
              </Pressable>
            );
          })}
        </View>

        <View style={styles.section}>
          <View style={styles.title}>
            <View style={styles.titleRow}>
              <Image source={require('../../assets/images/anatomical-heart.png')} style={styles.image} />
              <CustomText style={styles.titleText}>심박수</CustomText>
            </View>
          </View>

          <View style={styles.content}>
            <CustomText style={styles.contentText}>평균 <CustomText style={styles.contentTextWeighted}>{averageHeartRate}</CustomText> bpm</CustomText>
            <View style={styles.contentGraph}>
              <CustomText style={styles.contentGraphTitle}>시간대별 심박수 변화</CustomText>

              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <LineChart
                  data={{
                    labels: displayLabels,
                    datasets: [
                      {
                        data: alignedHeartRate,
                      },
                    ],
                  }}
                  width={timeLabels.length * 25}
                  height={220}
                  yAxisSuffix=" bpm"
                  yAxisInterval={6}
                  chartConfig={{
                    // backgroundGradientFrom: '#F8F7F6',
                    // backgroundGradientTo: '#F8F7F6',
                    backgroundGradientFrom: '#F2EFED',
                    backgroundGradientTo: '#F2EFED',
                    decimalPlaces: 0,
                    color: (opacity = 1) => `rgba(122, 102, 82, ${opacity})`,
                    labelColor: (opacity = 1) => `rgba(67, 66, 64, ${opacity})`,
                    propsForDots: { r: '3' },
                  }}
                  style={{ marginVertical: 16, marginHorizontal: 4, borderRadius: 16 }}
                />
              </ScrollView>
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.title}>
            <View style={styles.titleRow}>
              <Image source={require('../../assets/images/gym.png')} style={styles.image} />
              <CustomText style={styles.titleText}>활동</CustomText>
            </View>
          </View>

          <View style={styles.content}>
            {totalExerciseDurationHr > 0 ? (
            <CustomText style={styles.contentText}>총 <CustomText style={styles.contentTextWeighted}>{totalExerciseDurationHr}</CustomText> 시간 <CustomText style={styles.contentTextWeighted}>{totalExerciseDurationMin}</CustomText> 분 / 총 <CustomText style={styles.contentTextWeighted}>{totalCalorie.toLocaleString()}</CustomText> 칼로리</CustomText>
            ) : (
            <CustomText style={styles.contentText}>총 <CustomText style={styles.contentTextWeighted}>{totalExerciseDurationMin}</CustomText> 분 / 총 <CustomText style={styles.contentTextWeighted}>{totalCalorie.toLocaleString()}</CustomText> 칼로리</CustomText>
            )}

            <View style={styles.contentGraph}>
              <CustomText style={styles.contentGraphTitle}>일일 활동 로그</CustomText>

              {exerciseLogs.map((log, index) => (
                <View key={index} style={styles.log}>
                  {log.type === 1001 ? <Image source={require('../../assets/images/walking.png')} style={styles.logImage} />
                    : log.type === 1002 ? <Image source={require('../../assets/images/user-fast-running.png')} style={styles.logImage} />
                    : log.type === 2001 ? <Image source={require('../../assets/images/baseball-player.png')} style={styles.logImage} />
                    : log.type === 3001 ? <Image source={require('../../assets/images/golfer.png')} style={styles.logImage} />
                    : log.type === 4004 ? <Image source={require('../../assets/images/football-player.png')} style={styles.logImage} />
                    : log.type === 11007 ? <Image source={require('../../assets/images/bicycle.png')} style={styles.logImage} />
                    : <Image source={require('../../assets/images/hiking.png')} style={styles.logImage} />}
                  <View style={styles.left}>
                    <CustomText>
                    {log.type === 1001 ? '걷기'
                    : log.type === 1002 ? '뛰기'
                    : log.type === 2001 ? '야구'
                    : log.type === 3001 ? '골프'
                    : log.type === 4004 ? '축구'
                    : log.type === 11007 ? '자전거 타기'
                    : '하이킹'}
                    </CustomText>
                    <CustomText>{formatToHourMinute(log.start_time)}~{formatToHourMinute(log.end_time)} ({Math.floor(log.duration / 1000 / 60 / 60)}:{String((log.duration / 1000 / 60).toFixed(0)).padStart(2, '0')})</CustomText>
                  </View>
                  <View style={styles.right}>
                    <CustomText>{log.average_heart_rate}bpm</CustomText>
                    <CustomText>{log.calorie}kcal</CustomText>
                  </View>
                </View>
              ))}
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.title}>
            <TouchableOpacity style={styles.titleRow}>
              <Image source={require('../../assets/images/walking.png')} style={styles.walkImage} />
              <CustomText style={styles.titleText}>걷기</CustomText>
            </TouchableOpacity>
          </View>

          <View style={styles.content}>
            <CustomText style={styles.contentText}>총 <CustomText style={styles.contentTextWeighted}>{totalStepCount.toLocaleString()}</CustomText> 걸음 / 총 <CustomText style={styles.contentTextWeighted}>{totalDistanceKm}.{totalDistanceM}</CustomText> km</CustomText>

            <View style={styles.contentGraph}>
              <CustomText style={styles.contentGraphTitle}>시간대별 걸음수 변화</CustomText>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <BarChart
                  data={{
                    labels: displayedLabels,
                    datasets: [
                      {
                        data: stepCounts,
                      },
                    ],
                  }}
                  width={Math.max(SCREEN_WIDTH, stepCounts.length * 20)}
                  height={220}
                  fromZero={true}
                  yAxisLabel=""
                  yAxisSuffix=" 걸음                         "
                  chartConfig={{
                    // backgroundGradientFrom: '#F8F7F6',
                    // backgroundGradientTo: '#F8F7F6',
                    backgroundGradientFrom: '#F2EFED',
                    backgroundGradientTo: '#F2EFED',
                    decimalPlaces: 0,
                    color: (opacity = 1) => `rgba(122, 102, 82, ${opacity})`,
                    labelColor: (opacity = 1) => `rgba(67, 66, 64, ${opacity})`,
                    fillShadowGradientOpacity: 1,
                    barPercentage: 0.3,
                    formatTopBarValue: (val) => (val === 0 ? '' : `${val}`),
                  }}
                  yAxisInterval={1}
                  showBarTops={false}
                  withInnerLines={true}
                  showValuesOnTopOfBars={true}
                  segments={5}
                  style={{ marginLeft: -30 }}
                />
              </ScrollView>
            </View>

            <View style={styles.gap} />

            <View style={styles.contentGraph}>
              <CustomText style={styles.contentGraphTitle}>시간대별 이동거리 변화</CustomText>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <BarChart
                  data={{
                    labels: displayedLabels,
                    datasets: [
                      {
                        data: roundedData,
                      },
                    ],
                  }}
                  width={Math.max(SCREEN_WIDTH, stepCounts.length * 20)}
                  height={220}
                  fromZero={true}
                  yAxisLabel=""
                  yAxisSuffix=" 걸음                         "
                  chartConfig={{
                    // backgroundGradientFrom: '#F8F7F6',
                    // backgroundGradientTo: '#F8F7F6',
                    backgroundGradientFrom: '#F2EFED',
                    backgroundGradientTo: '#F2EFED',
                    decimalPlaces: 0,
                    color: (opacity = 1) => `rgba(122, 102, 82, ${opacity})`,
                    labelColor: (opacity = 1) => `rgba(67, 66, 64, ${opacity})`,
                    fillShadowGradientOpacity: 1,
                    barPercentage: 0.3,
                    formatTopBarValue: (val) => (val === 0 ? '' : `${val}`),
                  }}
                  yAxisInterval={1}
                  showBarTops={false}
                  withInnerLines={true}
                  showValuesOnTopOfBars={true}
                  segments={5}
                  style={{ marginLeft: -30 }}
                />
              </ScrollView>
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.title}>
            <View style={styles.titleRow}>
              <Image source={require('../../assets/images/snooze.png')} style={styles.sleepImage} />
              <CustomText style={styles.titleText}>수면</CustomText>
            </View>

            <View style={styles.content}>
              <CustomText style={styles.contentText}>총 <CustomText style={styles.contentTextWeighted}>{totalSleepDurationHour}</CustomText> 시간 <CustomText style={styles.contentTextWeighted}>{totalSleepDurationMin}</CustomText> 분 / 효율 <CustomText style={styles.contentTextWeighted}>{efficiency}</CustomText> 점</CustomText>

              <View style={styles.contentGraph}>
                <CustomText style={styles.contentGraphTitle}>시간대별 수면단계 변화</CustomText>
                <View style={styles.barContainer}>
                  {sleepStages.map((stage, index) => {
                    const startMin = timeToMinutes(stage.start);
                    const endMin = timeToMinutes(stage.end);
                    const offset = startMin - baseStart;
                    const duration = endMin - startMin;

                    return (
                      <View
                        key={index}
                        style={{ position: 'absolute', left: `${(offset / totalMinutesforSleep) * 100}%`, width: `${(duration / totalMinutesforSleep) * 100}%`, height: 20, backgroundColor: stage.color }}
                      />
                    );
                  })}
                </View>

                <View style={styles.timeLabels}>
                  {['22시', '0시', '2시', '4시', '6시', '8시'].map((label, index) => (
                    <CustomText key={index} style={styles.timeLabel}>{label}</CustomText>
                  ))}
                </View>

                <View style={styles.legend}>
                  {aggregatedStageList.map((stage, index) => (
                    <View key={index} style={styles.legendRow}>
                      <View style={[styles.colorDot, {
                        backgroundColor: stage.color,
                        width: `${(stage.duration / totalDuration) * 100}%`,
                      }]} />
                      <CustomText style={[styles.percentageText, { color: stage.color }]}>{Math.round((stage.duration / totalDuration) * 100)}%</CustomText>
                      <CustomText style={styles.legendText}>{stage.label} ({Math.floor(stage.duration / 60)}:{String(stage.duration % 60).padStart(2, '0')})</CustomText>
                    </View>
                  ))}
                </View>
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
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
  sliderContainer: {
    flexDirection: 'row',
    backgroundColor: '#F2EFED',
    borderRadius: 12,
    overflow: 'hidden',
  },
  slider: {
    position: 'absolute',
    width: TAB_WIDTH,
    height: '100%',
    backgroundColor: '#D6CCC2',
    borderRadius: 12,
    zIndex: 0,
  },
  tab: {
    width: TAB_WIDTH,
    paddingVertical: 12,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
  },
  date: {
    fontSize: 18,
    marginTop: 2,
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
    // justifyContent: 'space-between',
    marginBottom: 10,
  },
  titleText: {
    fontSize: 24,
  },
  image: {
    width: 17,
    height: 17,
    marginRight: 5,
  },
  walkImage: {
    width: 19,
    height: 19,
    marginRight: 4,
  },
  sleepImage: {
    width: 17,
    height: 17,
    marginRight: 5,
  },
  content: {
    paddingHorizontal: 10,
  },
  contentText: {
    marginBottom: 10,
    marginLeft: 7,
    fontSize: 18,
    color: '#434240',
  },
  contentTextWeighted: {
    fontSize: 24,
  },
  contentGraph: {
    // paddingLeft: 7,
    paddingHorizontal: 7,
    borderRadius: 10,
    // backgroundColor: '#F8F7F6',
    backgroundColor: '#F2EFED',
  },
  contentGraphTitle: {
    marginTop: 15,
    marginBottom: 10,
    marginHorizontal: 8,
    fontSize: 20,
  },
  log: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
    // marginLeft: 8,
    marginHorizontal: 8,
    // marginRight: 15,
    marginRight: 10,
    padding: 10,
    borderRadius: 10,
    // backgroundColor: '#EBE7E2',
    backgroundColor: '#D6CCC2',
  },
  gap: {
    height: 15,
  },
  logImage: {
    width: 20,
    height: 20,
  },
  left: {
    marginLeft: 10,
  },
  right: {
    marginLeft: 'auto',
  },
  barContainer: {
    height: 20,
    backgroundColor: '#E4E4E4',
    // backgroundColor: '#F2EFED',
    borderRadius: 4,
    marginBottom: 8,
    marginHorizontal: 8,
    position: 'relative',
    overflow: 'hidden',
  },
  timeLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
    marginHorizontal: 8,
  },
  timeLabel: {
    fontSize: 12,
    color: '#888',
  },
  legend: {
    marginTop: 12,
    marginHorizontal: 8,
  },
  legendRow: {
    width: '70%',
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  colorDot: {
    // textAlign: 'center',
    justifyContent: 'center',
    width: 12,
    height: 20,
    borderRadius: 6,
    marginRight: 4,
  },
  percentageText: {
    marginRight: 4,
    fontSize: 14,
    // color: '#FFFFFF',
  },
  legendText: {
    // marginLeft: 5,
    fontSize: 16,
    color: '#434240',
  },
});

export default BiometricOverviewScreen;