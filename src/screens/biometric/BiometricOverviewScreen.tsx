// import React from 'react';
// import { View, SafeAreaView, TouchableOpacity, StyleSheet } from 'react-native';
// import { useNavigation } from '@react-navigation/native';
// import { StackNavigationProp } from '@react-navigation/stack';
// import { RootStackParamList } from '../../types/navigationTypes.ts';
// import Icon from '../../components/Icon.tsx';

// const BiometricOverviewScreen = () => {
//   type Navigation = StackNavigationProp<RootStackParamList, 'BiometricOverview'>;
//   const navigation = useNavigation<Navigation>();

//   return (
//     <View style={styles.container}>
//       <SafeAreaView style={styles.safeAreaWrapper}>
//         <TouchableOpacity onPress={() => navigation.navigate('Home')}>
//           <Icon name="chevron-back" size={16} />
//         </TouchableOpacity>
//       </SafeAreaView>

//     </View>
//   );
// };

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     paddingVertical: 10,
//     paddingHorizontal: 16,
//     backgroundColor: '#FFFFFF',
//   },
//   safeAreaWrapper: {
//     width: 70,
//     paddingVertical: 16,
//     paddingHorizontal: 10,
//   },
// });

// export default BiometricOverviewScreen;

import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, Dimensions, SafeAreaView, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes';
import Icon from '../../components/Icon';
import { BarChart } from 'react-native-chart-kit';

const screenWidth = Dimensions.get('window').width;

const BiometricOverviewScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'BiometricOverview'>;
  const navigation = useNavigation<Navigation>();

  const barChartConfig = {
    backgroundColor: '#ffffff',
    backgroundGradientFrom: '#ffffff',
    backgroundGradientTo: '#ffffff',
    fillShadowGradient: '#ED6D4E',
    fillShadowGradientOpacity: 1,
    color: () => '#ED6D4E',
    labelColor: () => '#999999',
    barPercentage: 0.5,
    decimalPlaces: 0,
  };

  const [selectedDate, setSelectedDate] = useState<'그제' | '어제' | '오늘'>('오늘');

  const dateOptions: ('그제' | '어제' | '오늘')[] = ['그제', '어제', '오늘'];

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeAreaWrapper}>
        <TouchableOpacity onPress={() => navigation.navigate('Home')}>
          <Icon name="chevron-back" size={16} />
        </TouchableOpacity>
      </SafeAreaView>

      <View style={styles.tabContainer}>
  {dateOptions.map((option) => (
    <TouchableOpacity
      key={option}
      style={[
        styles.tabButton,
        selectedDate === option && styles.tabButtonActive,
      ]}
      onPress={() => setSelectedDate(option)}
    >
      <Text
        style={[
          styles.tabButtonText,
          selectedDate === option && styles.tabButtonTextActive,
        ]}
      >
        {option}
      </Text>
    </TouchableOpacity>
  ))}
</View>

      <ScrollView showsVerticalScrollIndicator={false} overScrollMode="never">
        {/* 날짜 선택 바 생략 - 필요 시 추가 가능 */}

        {/* 걸음 수 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>걸음 수</Text>
          <Text style={styles.sectionSummary}>총 6,855 걸음</Text>
          <BarChart
            data={{
              labels: ['6시', '8시', '10시', '12시', '14시', '16시', '18시'],
              datasets: [{ data: [300, 800, 1200, 1500, 1300, 700, 100] }],
            }}
            width={screenWidth - 32}
            height={180}
            yAxisLabel="" // ← 여기를 추가
            yAxisSuffix="" // ← 여기를 추가
            chartConfig={barChartConfig}
            withInnerLines={false}
            showValuesOnTopOfBars
            fromZero
          />
        </View>

        {/* 활동 시간 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>활동 시간</Text>
          <Text style={styles.sectionSummary}>총 3시간 30분</Text>
          <BarChart
            data={{
              labels: ['6시', '8시', '10시', '12시', '14시', '16시', '18시'],
              datasets: [{ data: [200, 600, 1100, 1000, 900, 800, 300] }],
            }}
            width={screenWidth - 32}
            height={180}
            yAxisLabel="" // ← 여기를 추가
            yAxisSuffix="" // ← 여기를 추가
            chartConfig={barChartConfig}
            withInnerLines={false}
            showValuesOnTopOfBars
            fromZero
          />
        </View>

        {/* 활동 칼로리 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>활동 칼로리</Text>
          <Text style={styles.sectionSummary}>총 2,010 칼로리</Text>
          <BarChart
            data={{
              labels: ['6시', '8시', '10시', '12시', '14시', '16시', '18시'],
              datasets: [{ data: [200, 600, 1100, 1000, 900, 800, 300] }],
            }}
            width={screenWidth - 32}
            height={180}
            yAxisLabel="" // ← 여기를 추가
            yAxisSuffix="" // ← 여기를 추가
            chartConfig={barChartConfig}
            withInnerLines={false}
            showValuesOnTopOfBars
            fromZero
          />
        </View>

        {/* 수면 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>수면</Text>
          <Text style={styles.sectionSummary}>총 7시간 30분</Text>
          <View style={styles.sleepGraphPlaceholder}>
            <Text style={styles.sleepText}>[여기에 수면 상태 바 그래프 대체 가능]</Text>
          </View>
          <View style={styles.sleepDetailContainer}>
            <Text style={styles.sleepDetail}>얕은 수면 23:10~00:30</Text>
            <Text style={styles.sleepDetail}>깊은 수면 00:30~02:20</Text>
            <Text style={styles.sleepDetail}>렘 수면 02:20~04:50</Text>
            <Text style={styles.sleepDetail}>각성 상태 04:50~06:40</Text>
          </View>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 16,
    backgroundColor: '#FFFFFF',
  },
  safeAreaWrapper: {
    width: 70,
    paddingVertical: 16,
    paddingHorizontal: 10,
  },
  tabContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  tabButton: {
    flex: 1,
    paddingVertical: 10,
    marginHorizontal: 4,
    borderWidth: 1,
    borderColor: '#D6CCC2',
    borderRadius: 8,
    alignItems: 'center',
  },
  tabButtonActive: {
    backgroundColor: '#D6CCC2',
  },
  tabButtonText: {
    fontSize: 14,
    color: '#8D8D8D',
  },
  tabButtonTextActive: {
    color: '#000000',
    fontWeight: 'bold',
  },
  
  section: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333333',
    marginBottom: 4,
  },
  sectionSummary: {
    fontSize: 14,
    color: '#888888',
    marginBottom: 8,
  },
  sleepGraphPlaceholder: {
    height: 24,
    backgroundColor: '#eee',
    borderRadius: 4,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  sleepText: {
    fontSize: 12,
    color: '#999',
  },
  sleepDetailContainer: {
    gap: 2,
  },
  sleepDetail: {
    fontSize: 13,
    color: '#444',
  },
});

export default BiometricOverviewScreen;
