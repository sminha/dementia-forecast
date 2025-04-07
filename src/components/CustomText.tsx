import React from 'react';
import { Text, TextProps } from 'react-native';

interface CustomTextProps extends TextProps {
  weight?: 'regular' | 'bold';
}

const CustomText: React.FC<CustomTextProps> = ({ weight = 'regular', style, children, ...props }) => {
  const getFontFamily = () => {
    switch(weight) {
      case 'bold':
        return 'Noto Sans KR ExtraBold';
      case 'regular':
      default:
        return 'Noto Sans KR SemiBold';
    }
  };

  return (
    <Text style={[{ fontFamily: getFontFamily() }, style]} {...props}>
      {children}
    </Text>
  );
};

export default CustomText;