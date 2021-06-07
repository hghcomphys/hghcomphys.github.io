---
title: "Create a GSM Remote Alarm System"
categories:
  - Electronics
tags:
  - Smart Home
  - ESP32
  - C++
header:
  image: &image "https://github.com/hghcomphys/remote-alarm-system/blob/master/docs/esp32_remote_alarm.JPG?raw=true"
  caption: "ESP32 Remote Alarm System"
  teaser: *image
# link: https://github.com/hghcomphys/remote-alarm-system
# classes: wide
# toc: true
# toc_label: "Table of Contents"
# toc_icon: "cog"
# layout: splash
# classes:
#   - landing
#   - dark-theme
---

_GSM Remote Alarm System_ is an experimental implementation of a broader idea of how one can build a home alarm system.
This post shows my attempts to remotely control an alarm system using [SIM800L][sim800lref] GSM module which is connected to a [ESP32][esp32ref] microcontroller. 

My challenge was particularly on the code side on how to communicate and control the GSM module to send/receive SMS messages, and then using those messages in order to activate alarm mode or even control external devices such as a relay.

For more details and the sketch (C++ source code) please see 
[https://github.com/hghcomphys/weather-monitor-module](https://github.com/hghcomphys/weather-monitor-module)


<!-- References -->
[esp32ref]: https://en.wikipedia.org/wiki/NodeMCU
[sim800lref]: https://lastminuteengineers.com/sim800l-gsm-module-arduino-tutorial/
