import { NativeModules, Platform } from 'react-native';

const { ExternalStoragePermission } = NativeModules;

export const openExternalStoragePermission = () => {
  if (Platform.OS === 'android' && Platform.Version >= 30) {
    ExternalStoragePermission.openManageAllFilesPermission();
  }
};
