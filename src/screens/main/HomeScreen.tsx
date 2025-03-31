import React from 'react';
import { View, TouchableOpacity, Text, FlatList, StyleSheet } from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';

const HomeScreen = () => {
  const dementiaInfo = [
    { id: '1', text: '🧬  치매는 유전병일까요?' },
    { id: '2', text: '👵🏻  치매는 노인들만 걸리는 병일까요?' },
    { id: '3', text: '🤒  치매와 알츠하이머는 같은 병일까요?' },
  ];

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.loginContainer}>
        <Text style={styles.loginText}>로그인이 필요합니다.</Text>
        <Icon name="chevron-forward" size={16} color="gray" />
      </TouchableOpacity>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>치매 진단하기</Text>
        <View style={styles.card}>
          <TouchableOpacity style={styles.row}>
            <Text style={styles.cardText}>라이프스타일 입력</Text>
            <Text style={styles.status}>미완</Text>
            <Icon name="chevron-forward" size={16} color="gray" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.row}>
            <Text style={styles.cardText}>생체정보 입력</Text>
            <Text style={styles.status}>미완</Text>
            <Icon name="chevron-forward" size={16} color="gray" />
          </TouchableOpacity>
        </View>
        <TouchableOpacity style={styles.disabledButton} disabled>
          <Text style={styles.buttonText}>진단하기</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.section}>
        <View style={styles.row}>
          <Text style={styles.sectionTitle}>진단 결과 보기</Text>
          <TouchableOpacity>
            <Text style={styles.moreText}>더보기</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.resultBox}>
          <Text style={styles.resultText}>진단 결과가 없습니다.</Text>
        </View>
      </View>

      <View style={styles.section}>
        <View style={styles.row}>
          <Text style={styles.sectionTitle}>치매 알아보기</Text>
          <TouchableOpacity>
            <Text style={styles.moreText}>더보기</Text>
          </TouchableOpacity>
        </View>
        <FlatList 
          data={dementiaInfo}
          keyExtractor={(item) => item.id}
          renderItem={({item}) => {
            return (
              <TouchableOpacity style={styles.infoCard}>
                <Text style={styles.infoText}>{item.text}</Text>
              </TouchableOpacity>
            );
          }}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#FFFFFF',
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
  sectionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  card: {
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
    fontSize: 18,
    color: '#434240',
  },
  status: {
    fontSize: 12,
    color: '#868481',
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
    padding: 12,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  resultText: {
    fontSize: 18,
    color: '#434240',
  },
  infoCard: {
    alignItems: 'center',
    padding: 12,
    marginVertical: 8,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  infoText: {
    fontSize: 18,
    color: '#434240',
  },
});

export default HomeScreen;