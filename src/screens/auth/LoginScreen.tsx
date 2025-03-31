import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import Icon from '../../components/Icon.tsx';

const LoginScreen = () => {
  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.backButton}>
        <Icon name="chevron-back" size={30} type="Ionicons" />
      </TouchableOpacity>

      <View style={styles.textContainer}>
        <Text style={styles.mainText}>
          반가워요!{'\n'}
          가입 및 로그인을 진행해주세요.
        </Text>
      </View>

      <TouchableOpacity style={styles.kakaoBox}>
        <Icon name="chatbubble" size={24} color="#3C1E1E" type="FontAwesome6" style={styles.icon} />
        <Text style={styles.kakaoBoxText}>카카오로 계속하기</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.emailBox}>
        <Icon name="envelope" size={24} color="#FFFFFF" type="Ionicons" style={styles.icon} />
        <Text style={styles.emailBoxText}>이메일로 계속하기</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#FFFFFF',
  },
  backButton: {
    position: 'absolute',
    top: 90,
    left: 53,
  },
  textContainer: {
    alignItems: 'center',
    marginTop: 140,
    marginBottom: 30,
  },
  mainText: {
    fontSize: 24,
  },
  kakaoBox: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#FDE500',
  },
  emailBox: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#575553',
  },
  icon: {
    marginRight: 10,
  },
  kakaoBoxText: {
    fontSize: 18,
  },
  emailBoxText: {
    fontSize: 18,
    color: '#FFFFFF',
  },
});

export default LoginScreen;