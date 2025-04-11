import React, { useState } from 'react';
import { View, SafeAreaView, TouchableOpacity, Modal, TouchableWithoutFeedback, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const LoginScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Login'>;
  const navigation = useNavigation<Navigation>();

  const [modalVisible, setModalVisible] = useState(false);

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Icon name="chevron-back" size={16} />
        </TouchableOpacity>
      </SafeAreaView>

      <View style={styles.topArea}>
        <CustomText style={styles.mainText}>반가워요!{'\n'}가입 및 로그인을 진행해주세요. </CustomText>
      </View>

      <View style={styles.bottomArea}>
        <TouchableOpacity style={[styles.authBox, styles.kakaoBox]}>
          <Icon name="chatbubble" size={24} color="#3C1E1E" style={styles.icon} />
          <CustomText style={styles.kakaoBoxText}>카카오로 계속하기</CustomText>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.authBox, styles.emailBox]} onPress={() => setModalVisible(true)}>
          <Icon name="mail" size={24} color="#FFFFFF" style={styles.icon} />
          <CustomText style={styles.emailBoxText}>이메일로 계속하기</CustomText>
        </TouchableOpacity>
      </View>

      <Modal
        transparent={true}
        animationType="fade"
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <TouchableWithoutFeedback onPress={() => setModalVisible(false)}>
          <View style={styles.modalOverlay}>
            <TouchableWithoutFeedback>
              <View style={styles.modalContent}>
                <TouchableOpacity
                  style={[styles.authButton, styles.authButtonPrimary]}
                  onPress={() => {
                    navigation.navigate('EmailSignUp');
                    setModalVisible(false);}}
                >
                  <CustomText style={styles.authButtonText}>신규 회원 가입</CustomText>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.authButton, styles.authButtonSecondary]}
                  onPress={() => {
                    navigation.navigate('EmailLogin');
                    setModalVisible(false);
                  }}
                >
                  <CustomText style={styles.authButtonText}>기존 회원 로그인</CustomText>
                </TouchableOpacity>
              </View>
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
      </Modal>
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
  safeArea: {
    width: 70,
    paddingVertical: 16,
    paddingHorizontal: 10,
  },
  topArea: {
    flex: 2,
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  mainText: {
    fontSize: 24,
  },
  bottomArea: {
    flex: 4,
    gap: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  authBox: {
    position: 'relative',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'stretch',
    padding: 14,
    marginHorizontal: 20,
    borderRadius: 8,
  },
  kakaoBox: {
    backgroundColor: '#FDE500',
  },
  emailBox: {
    backgroundColor: '#575553',
  },
  icon: {
    position: 'absolute',
    left: 20,
  },
  kakaoBoxText: {
    fontSize: 18,
  },
  emailBoxText: {
    fontSize: 18,
    color: '#FFFFFF',
  },
  modalOverlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(69, 69, 69, 0.3)',
  },
  modalContent: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
    width: '100%',
    padding: 10,
    borderTopLeftRadius: 10,
    borderTopRightRadius: 10,
    backgroundColor: '#FFFFFF',
  },
  authButton: {
    flex: 1,
    alignItems: 'center',
    padding: 10,
    borderRadius: 10,
  },
  authButtonPrimary: {
    backgroundColor: '#F2EAE3',
  },
  authButtonSecondary: {
    backgroundColor: '#D6CCC2',
  },
  authButtonText: {
    fontSize: 18,
    color: '#575553',
  },
});

export default LoginScreen;