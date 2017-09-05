#!/usr/bin/env python
# encoding=utf-8
#----------------------------------------------------------------------------
# Dump the article in the last two months from ratedb
# For each group, the following features are extracted:
#   -- namespace P --
#   1) TT: top3 topic index
#----------------------------------------------------------------------------
#
#
import sys, MySQLdb, logging, datetime, re, os, signal, json, math, StringIO, time, struct
import random
sys.path.insert(0, ".")
from pyutil.program.log import logging_config
from pyutil.common import time_util
from ss_data.domain.const.article import ArticleComposition, ArticleCompositionMask, bits_set
from ss_data.pps.tools.group_attr_tool import GroupAttrToolOffline
import socket


class GroupContentDumper(object):
    def __init__(self):
        self.dump_days = 70

    def remove_tag(self, prefix, suffix, content):
        new_content = ''
        prev_pos = 0
        while True:
            pos = content.find(prefix, prev_pos)
            if pos < 0:
                new_content += content[prev_pos:]
                break
            new_content += content[prev_pos:pos]
            prev_pos = pos + len(prefix)
            pos = content.find(suffix, prev_pos)
            if pos < 0:
                logging.error('unmatched img tag')
                break
            prev_pos = pos + len(suffix)
        return new_content
        

    def clean_content(self, content):
        if content is None:
            return content
        content = content.replace('\r', '')
        content = content.replace('\n', '')
        content = content.replace('</div>', '')
        content = content.replace('<strong>', '')
        content = content.replace('</strong>', '')
        content = content.replace('<blockquote>', '')
        content = content.replace('</blockquote>', '')
        content = content.replace('<span>', '')
        content = content.replace('</span>', '')
        content = content.replace('<h1>', '')
        content = content.replace('</h1>', '')
        content = content.replace('<b>', '')
        content = content.replace('</b>', '')
        content = content.replace('<i>', '')
        content = content.replace('</i>', '')
        content = content.replace('<u>', '')
        content = content.replace('</u>', '')
        content = content.replace('<em>', '')
        content = content.replace('</em>', '')
        content = content.replace('<dd>', '')
        content = content.replace('</dd>', '')
        content = content.replace('<h2>', '')
        content = content.replace('</h2>', '')
        content = content.replace('<h3>', '')
        content = content.replace('</h3>', '')
        content = content.replace('</td>', '')
        content = content.replace('</tr>', '')
        content = content.replace('</ul>', '')
        content = content.replace('</li>', '')
        content = content.replace('</dt>', '')
        content = content.replace('</dl>', '')
        content = content.replace('</p>', '')
        content = content.replace('<br>', '')
        content = self.remove_tag('<p', '>', content)
        content = self.remove_tag('<dt', '>', content)
        content = self.remove_tag('<dl', '>', content)
        content = self.remove_tag('<table', '>', content)
        content = self.remove_tag('<tbody', '>', content)
        content = self.remove_tag('<th', '>', content)
        content = self.remove_tag('<div', '>', content)
        content = self.remove_tag('<img', '>', content)
        content = self.remove_tag('<ul', '>', content)
        content = self.remove_tag('<li', '>', content)
        content = self.remove_tag('<td', '>', content)
        content = self.remove_tag('<tr', '>', content)
        content = self.remove_tag('<table', '>', content)
        content = self.remove_tag('{!--', '--}', content)
        content = content.replace('</th>', '')
        content = content.replace('</table>', '')
        content = content.replace('</tbody>', '')
        return content 

    def process(self, file_name, todump_number):
        logging.info("begin to process")
        output = open(file_name, 'wb')
        start = time.time()
        dumped_groups = 0
        end_t = datetime.datetime.now()
        removed_count, short_title_cnt, short_content_cnt = 0, 0, 0
        while dumped_groups < todump_number:
            start_t = end_t - datetime.timedelta(hours=12)
            group_res = GroupAttrToolOffline().get_attr_by_time(start_t, end_t, set(["title","content","extra"]), time_type='create', recommend_status=1, caller="data.push.ngram")
            end_t = start_t
            for gid in group_res:
                if 'extra' in group_res[gid]:
                    try:
                        res_dict = json.loads(group_res[gid]['extra'],'{}')
                        composition = res_dict['composition']
                        if not (composition == 0 or composition == ArticleComposition.only_text_image):
                            removed_count += 1
                            continue
                    except:
                        pass
                if len(group_res[gid].get('title','')) < 18:
                    short_title_cnt += 1
                    continue
                content = self.clean_content(group_res[gid].get('content',''))
                if len(content) < 150:
                    short_content_cnt += 1
                    continue
                content = content.replace('\t',' ')
                if random.randint(1,10000) > 2000:
                    continue
                output.write('%s\t%s\t%s\n' % (gid, group_res.get('title','').replace('\t',' '), content) )
                dumped_groups += 1
                if dumped_groups == todump_number:
                    break
        output.close()
        logging.info(end_t)
        logging.info('removed_count %s\n' % removed_count)
        logging.info('short_title_cnt %s\n' % short_title_cnt)
        logging.info('short_content_cnt %s\n' % short_content_cnt)
        logging.info("save into file time cost %.4fs", time.time()-start)
        logging.info("end of process")

    def dump(self, file_name):
        gid_label = {}
        output = open(file_name, 'w')
        input = open('push_raw.dat')
        for line in input:
            tokens = line.strip().split()
            gid_label[tokens[0]] = tokens[1]
            group_res = GroupAttrToolOffline().get_attr_by_gids([tokens[0]], set(['title','content']), caller="data.push.ngram")
            if long([tokens[0]]) not in group_res:
                continue
            output.write('LABEL:%s\n' % tokens[1])
            output.write('GID:%s\n' % tokens[0])
            output.write('TITLE:%s\n' % group_res[long(tokens[0])].get('title',''))
            content = self.clean_content(group_res[long(tokens[0])].get('content',''))
            output.write('CONTENT:%s\n\n' % content)
        output.close()
def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if len(sys.argv) != 3:
        print 'Usage:\n\tpython %s out_file dump_number'
        sys.exit(1)
    logging_config(log_file="log.dump")
    gfg = GroupContentDumper()
    gfg.process(sys.argv[1], int(sys.argv[2]))

if __name__ == '__main__':
    main()


