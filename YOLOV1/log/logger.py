#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 下午1:01
# @Author  : jyl
# @File    : logger.py
import logging.config

logging.config.fileConfig('./logging.conf')
yolologger = logging.getLogger('YoloV1Logger')

yolologger.debug('debug message')
yolologger.info(f'{"#"*80}')


