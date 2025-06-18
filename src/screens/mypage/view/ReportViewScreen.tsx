import React, { useState, useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, StyleSheet } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation, RouteProp, useRoute } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../../types/navigationTypes.ts';
import { AppDispatch, RootState } from '../../redux/store.ts';
import { fetchReport } from '../../../redux/slices/reportSlice.ts';
import CustomText from '../../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';

const ReportViewScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const route = useRoute<RouteProp<RootStackParamList, 'ReportView'>>();
  const { from } = route.params;

  const dispatch = useDispatch<AppDispatch>();

  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // console.log('가져온 리포트: ', reports[1]);

  useEffect(() => {
    const loadReports = async () => {
      try {
        const token = await AsyncStorage.getItem('accessToken');
        const createDateJson = await AsyncStorage.getItem('createDate');
        const dateList: number[] = createDateJson ? JSON.parse(createDateJson) : [];

        const reportResults = await Promise.all(
          dateList.map(async (date) => {
            const result = await dispatch(fetchReport({ token, date }));
            if (fetchReport.fulfilled.match(result)) {
              return { date, data: result.payload };
            } else {
              return null;
            }
          })
        );

        const validReports = reportResults.filter((item) => item !== null);
        setReports(validReports);
      } catch (e) {
        console.error('보고서 불러오기 실패', e);
      } finally {
        setLoading(false);
      }
    };

    loadReports();
  }, []);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          <TouchableOpacity style={styles.backContainer} onPress={() => from === 'Home' || from === 'ReportResult' ? navigation.replace('Home') : navigation.replace('Mypage')}>
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity>
          <View style={styles.titleWrapper}>
            <CustomText style={styles.loginText}>진단 결과</CustomText>
          </View>
        </View>

        <View style={styles.section}>
          {reports.length > 0 ? (
            <View style={styles.list}>
              {reports
                .sort((a, b) => b.date - a.date)
                .slice(0, 4)
                .map((report, index) => (
                  <TouchableOpacity
                    key={index}
                    style={styles.listRow}
                    onPress={() => navigation.navigate('ReportResult', { type: 'reportView', data: report.data })}
                  >
                    <CustomText style={styles.listText}>
                      {`${report.date.toString().slice(2, 4)}/${report.date.toString().slice(4, 6)}/${report.date.toString().slice(6, 8)}`}
                    </CustomText>
                    <View style={styles.result}>
                      {report.data.risk_score === null ? ''
                      : report.data.risk_score >= 80 ? (<CustomText style={styles.resultText}>☁️ 매우 흐림, {report.data.risk_score}%</CustomText>)
                      : report.data.risk_score >= 60 ? (<CustomText style={styles.resultText}>⛅ 흐림, {report.data.risk_score}%</CustomText>)
                      : report.data.risk_score >= 50 ? (<CustomText style={styles.resultText}>🌤️ 약간 흐림, {report.data.risk_score}%</CustomText>)
                      : (<CustomText style={styles.resultText}>☀️ 맑음, {report.data.risk_score}%</CustomText>)
                      }
                    </View>
                    <Icon name="chevron-forward" size={16} color="gray" />
                  </TouchableOpacity>
                ))}
            </View>
          ) : (
            <View style={{ marginTop: '100%', alignItems: 'center', justifyContent: 'center' }}>
              <CustomText style={{ fontSize: 20 }}>진단 결과가 없습니다.</CustomText>
            </View>
          )}
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
    // backgroundColor: 'red',
  },
  sectionSecondary: {
    marginTop: 30,
  },
  sectionTitle: {
    fontSize: 24,
    marginBottom: 8,
  },
  list: {
    paddingHorizontal: 5,
    paddingTop: 10,
  },
  listText: {
    fontSize: 18,
    color: '#434240',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingRight: 5,
  },
  listRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    // marginBottom: 20,
    marginBottom: 30,
    position: 'relative',
  },
  result: {
    position: 'absolute',
    left: 0,
    right: 30,
    alignItems: 'flex-end',
    justifyContent: 'center',
    marginBottom: 3,
  },
  resultText: {
    fontSize: 18,
    color: '#434240',
  }
});

export default ReportViewScreen;