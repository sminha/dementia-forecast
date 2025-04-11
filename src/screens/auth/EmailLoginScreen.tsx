import React, { useState } from 'react';
import { View, SafeAreaView, TouchableOpacity, ScrollView, TextInput, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const EmailLoginScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'EmailLogin'>;
  const navigation = useNavigation<Navigation>();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [secureText, setSecureText] = useState(true);

  const handleFocus = (field: string) => {
    setFocusedField(field);
  };

  const handleBlur = () => {
    setFocusedField(null);
  };

  const isFormValid = () => {
    return email.trim() !== '' && password.trim() !== '';
  };

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeAreaWrapper}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Icon name="chevron-back" size={16} />
        </TouchableOpacity>
      </SafeAreaView>

      <View style={styles.headerContainer}>
        <CustomText style={styles.titleText}>이메일로 로그인하기</CustomText>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContentContainer} overScrollMode="never">
        <View style={styles.inputGroup}>
          <CustomText style={[styles.labelText, focusedField === 'email' && styles.labelTextFocused]}>이메일</CustomText>
          <View style={styles.inputFieldWrapper}>
            <TextInput
              style={[styles.inputField, focusedField === 'email' && styles.inputFieldFocused]}
              onFocus={() => handleFocus('email')}
              onBlur={() => handleBlur()}
              onChangeText={setEmail}
              value={email}
            />
            {email !== '' &&
            <TouchableOpacity onPress={() => setEmail('')} style={styles.clearButton}>
              <Icon name="close-circle" size={20} color="#B4B4B4" />
            </TouchableOpacity>
            }
          </View>
        </View>

        <View style={styles.inputGroup}>
          <CustomText style={[styles.labelText, focusedField === 'password' && styles.labelTextFocused]}>비밀번호</CustomText>
          <View style={styles.inputFieldWrapper}>
            <TextInput
              style={[styles.inputField, focusedField === 'password' && styles.inputFieldFocused]}
              onFocus={() => handleFocus('password')}
              onBlur={() => handleBlur()}
              onChangeText={setPassword}
              secureTextEntry={secureText}
              value={password}
            />
            {password !== '' && (
              <TouchableOpacity onPress={() => setSecureText(!secureText)} style={styles.clearButton}>
                <Icon name={secureText ? 'eye' : 'eye-off'} size={20} color="#B4B4B4" />
              </TouchableOpacity>
            )}
          </View>
        </View>
      </ScrollView>

      <View>
        <TouchableOpacity onPress={() => navigation.navigate('Home')} style={[styles.actionButton, isFormValid() ? styles.actionButtonEnabled : styles.actionButtonDisabled]} disabled={!isFormValid()}>
          <CustomText style={[styles.actionButtonText, isFormValid() ? styles.actionButtonTextEnabled : styles.actionButtonTextDisabled]}>로그인</CustomText>
        </TouchableOpacity>
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
  headerContainer: {
    paddingTop: 40,
    paddingBottom: 50,
    paddingHorizontal: 20,
  },
  backButton: {
    marginBottom: 10,
  },
  titleText: {
    fontSize: 24,
  },
  scrollContentContainer: {
    paddingHorizontal: 20,
  },
  inputGroup: {
    marginBottom: 40,
  },
  labelText: {
    fontSize: 18,
    color: '#202020',
  },
  labelTextFocused: {
    color: '#9F8473',
  },
  inputFieldWrapper: {
    position: 'relative',
    flexDirection: 'row',
    alignItems: 'center',
  },
  clearButton: {
    position: 'absolute',
    top: '50%',
    right: 5,
    transform: [{ translateY: -8 }],
  },
  inputField: {
    width: '100%',
    paddingRight: 35,
    marginTop: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    fontSize: 18,
  },
  inputFieldFocused: {
    borderBottomColor: '#9F8473',
  },
  chevronIcon: {
    marginLeft: 'auto',
  },
  actionButton: {
    alignItems: 'center',
    padding: 15,
    marginTop: 10,
    marginBottom: 5,
    borderRadius: 5,
  },
  actionButtonEnabled: {
    backgroundColor: '#D6CCC2',
  },
  actionButtonDisabled: {
    backgroundColor: '#F2EFED',
  },
  actionButtonText: {
    fontSize: 20,
  },
  actionButtonTextEnabled: {
    color: '#575553',
  },
  actionButtonTextDisabled: {
    color: '#B4B4B4',
  },
});

export default EmailLoginScreen;