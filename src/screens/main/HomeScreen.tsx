import React from 'react';
import { ScrollView, View, TouchableOpacity, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';

const HomeScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dementiaInfo = [
    { id: '1', text: '🧬  치매는 유전병일까요? ' },
    { id: '2', text: '👵🏻  치매는 노인들만 걸리는 병일까요? ' },
    { id: '3', text: '🤒  치매와 알츠하이머는 같은 병일까요? ' },
    // { id: '4', text: '💭  치매와 단순 건망증은 어떻게 구별하나요? ' },
    // { id: '5', text: '😄  치매는 완치가 가능한가요? ' },
  ];

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} overScrollMode="never">
        <TouchableOpacity style={styles.loginContainer} onPress={() => navigation.navigate('Login')}>
          <CustomText style={styles.loginText}>로그인이 필요합니다. </CustomText>
          <Icon name="chevron-forward" size={16} color="gray" />
        </TouchableOpacity>

        <View style={styles.section}>
          <CustomText weight="bold" style={styles.sectionTitle}>치매 진단하기</CustomText>
          <View style={styles.card}>
            <TouchableOpacity style={styles.row} onPress={() => navigation.navigate('LifestyleStart')}>
              <CustomText style={styles.cardText}>라이프스타일 입력</CustomText>
              <View style={styles.statusContainer}>
                <CustomText style={styles.status}>미완</CustomText>
                <Icon name="chevron-forward" size={16} color="gray" />
              </View>
            </TouchableOpacity>
            <TouchableOpacity style={styles.row} onPress={() => navigation.navigate('BiometricStart')}>
              <CustomText style={styles.cardText}>생체정보 입력</CustomText>
              <View style={styles.statusContainer}>
                <CustomText style={styles.status}>미완</CustomText>
                <Icon name="chevron-forward" size={16} color="gray" />
              </View>
            </TouchableOpacity>
          </View>
          <TouchableOpacity style={styles.disabledButton} disabled>
            <CustomText style={styles.buttonText}>진단하기</CustomText>
          </TouchableOpacity>
        </View>

        <View style={styles.sectionSecondary}>
          <View style={styles.row}>
            <CustomText weight="bold" style={styles.sectionTitle}>진단 결과 보기</CustomText>
            <TouchableOpacity>
              <CustomText style={styles.moreText}>더보기</CustomText>
            </TouchableOpacity>
          </View>
          <View style={styles.resultBox}>
            <CustomText style={styles.resultText}>진단 결과가 없습니다. </CustomText>
          </View>
        </View>

        <View style={styles.sectionSecondary}>
          <View style={styles.row}>
            <CustomText weight="bold" style={styles.sectionTitle}>치매 알아보기</CustomText>
            <TouchableOpacity>
              <CustomText style={styles.moreText}>더보기</CustomText>
            </TouchableOpacity>
          </View>
          {dementiaInfo.map((item) => (
            <TouchableOpacity key={item.id} style={styles.infoCard}>
              <CustomText style={styles.infoText}>{item.text}</CustomText>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.sectionSecondary}>
          <View style={styles.row}>
            <CustomText weight="bold" style={styles.sectionTitle}>치매 예방하기</CustomText>
          </View>
            <TouchableOpacity style={styles.preventionCard}>
              <CustomText style={styles.infoText}>치매 예방 체조</CustomText>
            </TouchableOpacity>
            <TouchableOpacity style={styles.preventionCard}>
              <CustomText style={styles.infoText}>치매 예방 체조</CustomText>
            </TouchableOpacity>
            <TouchableOpacity style={styles.preventionCard}>
              <CustomText style={styles.infoText}>치매 예방 체조</CustomText>
            </TouchableOpacity>
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
  loginContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 10,
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
  card: {
    marginTop: 8,
    padding: 12,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  cardText: {
    paddingLeft: 6,
    fontSize: 18,
    color: '#6C6C6B',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  status: {
    fontSize: 12,
    color: '#9F9E9B',
  },
  disabledButton: {
    alignItems: 'center',
    padding: 10,
    marginTop: 10,
    borderRadius: 10,
    backgroundColor: '#F2EFED',
  },
  buttonText: {
    fontSize: 18,
    color: '#B4B4B4',
  },
  moreText: {
    fontSize: 16,
    color: '#868481',
  },
  resultBox: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 50,
    paddingHorizontal: 12,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  resultText: {
    fontSize: 18,
    color: '#6C6C6B',
  },
  infoCard: {
    padding: 12,
    marginVertical: 5,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  infoText: {
    paddingLeft: 6,
    fontSize: 18,
    color: '#6C6C6B',
  },
  preventionCard: {
    paddingLeft: 12,
    paddingVertical: 20,
    marginVertical: 5,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
});

export default HomeScreen;